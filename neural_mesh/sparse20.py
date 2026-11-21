"""Twenty *locally training* specialists, sparse five-peer distillation, durable state.

Every stage-1 / stage-2 node runs in an independent GitHub runner process.
Only compact forecast probes and validation reliability cross node boundaries.
The supervised learner uses stdlib-only small numerical adapters, not LLMs.
No latent feature coordinates are assumed to align between independently
trained specialists: shared messages are 2-D forecast vectors in world units.

Stage 2 is a paired experiment: control and five-peer branches start from
identical models, receive identical local labeled training samples and have
the same local SGD budget. Peer distillation uses *unlabelled* probes.
"""
from __future__ import annotations
import argparse
import copy
import math
import random
import statistics
from pathlib import Path

from neural_mesh import mesh, swarm20

SCHEMA = "neuralmesh-sparse-learning-v1"
NODES = swarm20.NODES
ROLES = swarm20.ROLES
REGIMES = swarm20.SCENES
PEERS = 5
PROBES = 32
TEST_SEEDS = (726031, 726037, 726043)
SCENES_PER_TEST = 160
GENERATION_CAP = 128
ROLE_FOCUS = {
    "position": "linear",
    "velocity": "bounce",
    "acceleration": "accelerating",
    "curvature": "turning",
    "stabilizer": "jerk",
}


def target_motion(seed, index, regime):
    """Return t0..t4: one common target for all sensors."""
    r = random.Random(swarm20.stable_seed("motion-v1",seed,index,regime))
    x,y = r.uniform(-2,2),r.uniform(-2,2)
    vx,vy = r.uniform(-.85,.85),r.uniform(-.85,.85)
    ax,ay = r.uniform(-.15,.15),r.uniform(-.15,.15)
    jx,jy = r.uniform(-.065,.065),r.uniform(-.065,.065)
    if regime=="linear":
        return [[x+vx*t,y+vy*t] for t in range(5)]
    if regime=="accelerating":
        return [[x+vx*t+.5*ax*t*t,y+vy*t+.5*ay*t*t] for t in range(5)]
    if regime=="jerk":
        return [[x+vx*t+.5*ax*t*t+jx*t**3/6,
                 y+vy*t+.5*ay*t*t+jy*t**3/6] for t in range(5)]
    if regime=="turning":
        a,omega,radius=r.uniform(-3.14,3.14),r.uniform(.14,.54),r.uniform(.3,.9)
        return [[x+.2*vx*t+radius*(math.cos(a+omega*t)-math.cos(a)),
                 y+.2*vy*t+radius*(math.sin(a+omega*t)-math.sin(a))]
                for t in range(5)]
    if regime=="bounce":
        return [[x+vx*(t if t<=2 else 4-t),y+vy*t] for t in range(5)]
    raise ValueError("unsupported physics")


def observed(seed,index,regime,node):
    points=target_motion(seed,index,regime)
    sensor_seed=swarm20.stable_seed("sensor-v1",seed,index,regime,node)
    return swarm20.sensor(points,node,random.Random(sensor_seed)),points[4]


def _initial(node,parent=None):
    role,replica=swarm20.identity(node)
    if parent is not None:
        _check_model(parent)
        return copy.deepcopy(parent)
    masks=swarm20.ROLE_MASKS[role]
    scale=masks[0]*(1+.04*(replica-1.5))
    w=[[0.]*swarm20.FEATURES for _ in (0,1)]
    w[0][0]=1./scale
    w[1][1]=1./scale
    return w


def _check_model(w):
    if (not isinstance(w,list) or len(w)!=2 or
        any(not isinstance(row,list) or len(row)!=swarm20.FEATURES
            for row in w) or
        any(not isinstance(x,(int,float)) or not math.isfinite(x) or
            abs(x)>4 for row in w for x in row)):
        raise ValueError("invalid persisted model")


def _update(w,node,track,label,rate):
    z=swarm20.features(track,node)
    p=swarm20.predict(track,node,w)
    for j in (0,1):
        residual=max(-3.,min(3.,p[j]-label[j]))
        for k in range(swarm20.FEATURES):
            w[j][k]=max(-4.,min(4.,w[j][k]-rate*residual*z[k]))


def _specialized_regime(seed,node,i,wave):
    r=random.Random(swarm20.stable_seed("local-regime",seed,node,wave,i))
    focus=ROLE_FOCUS[swarm20.identity(node)[0]]
    return focus if r.random()<.62 else REGIMES[r.randrange(len(REGIMES))]


def _learn_local(node,w,seed,wave,count):
    for i in range(count):
        mode=_specialized_regime(seed,node,i,wave)
        track,label=observed(seed+wave*10007,i,mode,node)
        _update(w,node,track,label,.009 if wave==1 else .0055)
    return w


def _calibration(node,w,seed):
    errors={r:[] for r in REGIMES}
    for i in range(150):
        mode=REGIMES[i%len(REGIMES)]
        track,label=observed(seed+38017,i,mode,node)
        pred=swarm20.predict(track,node,w)
        errors[mode].append(mesh.squared_error(pred,label))
    return {mode:statistics.fmean(errors[mode]) for mode in REGIMES}


def _probe(node,w,seed):
    result=[]
    for i in range(PROBES):
        mode=REGIMES[i%len(REGIMES)]
        track,_=observed(seed+92179,i,mode,node)
        result.append(swarm20.predict(track,node,w))
    return result


def _receipt(payload):
    return {"payload":payload,"sha256":mesh.digest(payload)}


def _verified(wrapper):
    if (not isinstance(wrapper,dict) or
        wrapper.get("sha256") != mesh.digest(wrapper.get("payload"))):
        raise ValueError("invalid node/model digest")
    return wrapper["payload"]


def _restore(parent_file):
    if not parent_file: return None
    payload=_verified(mesh.load(parent_file))
    if (payload.get("schema")!=SCHEMA or
        not isinstance(payload.get("generation"),int) or
        not 1<=payload["generation"]<GENERATION_CAP or
        set(payload.get("nodes",{}))!=set(NODES)):
        raise ValueError("invalid durable checkpoint or node population")
    for node in NODES:
        _check_model(payload["nodes"][node])
    return payload


def prepare(seed,out,parent=None):
    if not isinstance(seed,int) or seed<0 or seed>=2**31:
        raise ValueError("invalid run seed")
    previous=_restore(parent)
    generation=previous["generation"]+1 if previous else 1
    obj={"schema":SCHEMA,"seed":seed,"generation":generation,
         "parent_sha256":mesh.load(parent)["sha256"] if parent else None,
         "nodes":previous["nodes"] if previous else None,
         "train_labeled_scenes_per_node":360,
         "update_labeled_scenes_per_node":180,
         "calibration_scenes_per_node":150,
         "unlabelled_shared_probes":PROBES,
         "test_seeds":list(TEST_SEEDS)}
    mesh.save(Path(out)/"experiment.json",obj)
    return obj


def first(node,config,parent=None):
    if node not in NODES or config.get("schema")!=SCHEMA:
        raise ValueError("invalid node or run config")
    previous=_restore(parent)
    if (bool(previous)!=(config.get("parent_sha256") is not None) or
        (previous and
         (config["parent_sha256"]!=mesh.load(parent)["sha256"] or
          config["generation"]!=previous["generation"]+1))):
        raise ValueError("mismatched historical checkpoint")
    w=_initial(node,previous["nodes"][node] if previous else None)
    start=mesh.digest(w)
    _learn_local(node,w,config["seed"],1,config["train_labeled_scenes_per_node"])
    payload={"schema":SCHEMA,"stage":1,"node":node,
             "seed":config["seed"],"generation":config["generation"],
             "config_sha256":mesh.digest(config),
             "initial_model_sha256":start,
             "trained_model":w,"calibration":_calibration(node,w,config["seed"]),
             "probes":_probe(node,w,config["seed"]),
             "local_labeled_training":config["train_labeled_scenes_per_node"]}
    return _receipt(payload)


def _read_receipts(folder,stage):
    paths=list(Path(folder).glob("node-*.json"))
    if len(paths)!=len(NODES):
        raise ValueError(f"expected 20 local learning receipts, got {len(paths)}")
    found={}
    for p in paths:
        x=_verified(mesh.load(p))
        node=x.get("node")
        if (x.get("stage")!=stage or node not in NODES or
            p.name!=f"node-{node}.json" or node in found):
            raise ValueError("missing, duplicate or spoofed learning receipt")
        found[node]=x
    if set(found)!=set(NODES):
        raise ValueError("incomplete node population")
    return found


def aggregate_first(config,folder):
    records=_read_receipts(folder,1)
    for node,r in records.items():
        _check_model(r.get("trained_model"))
        if (r.get("schema")!=SCHEMA or r.get("seed")!=config["seed"] or
            r.get("generation")!=config["generation"] or
            r.get("config_sha256")!=mesh.digest(config) or
            set(r.get("calibration",{}))!=set(REGIMES) or
            len(r.get("probes",[]))!=PROBES or
            any(len(z)!=2 or any(not math.isfinite(a) for a in z)
                for z in r["probes"]) or
            any(v<0 or not math.isfinite(v)
                for v in r["calibration"].values())):
            raise ValueError(f"invalid training or shared probe receipt: {node}")
    routing={}
    for receiver in NODES:
        ownrole=swarm20.identity(receiver)[0]
        chosen=[]
        # Best one from each other competency family plus one own-family peer.
        for role in ROLES:
            others=[n for n in NODES if n!=receiver and
                    swarm20.identity(n)[0]==role]
            if not others: continue
            selected=min(others,key=lambda n:statistics.fmean(
                records[n]["calibration"].values()))
            chosen.append(selected)
        if len(chosen)!=PEERS or len(set(chosen))!=PEERS:
            raise ValueError("wrong sparse routing degree")
        scores={n:1./(1e-4+statistics.fmean(
                    records[n]["calibration"].values())) for n in chosen}
        norm=sum(scores.values())
        routing[receiver]={n:scores[n]/norm for n in chosen}
        if receiver in routing[receiver]:
            raise ValueError("self feedback disallowed")
    payload={"schema":SCHEMA,"stage":"sparse-route",
             "seed":config["seed"],"generation":config["generation"],
             "config_sha256":mesh.digest(config),
             "node_count":len(NODES),"peers_per_node":PEERS,
             "routing":routing,
             "probes":{n:records[n]["probes"] for n in NODES},
             "model_sha256":{n:mesh.digest(records[n]["trained_model"])
                             for n in NODES},
             "receipt_digest":mesh.digest({
                 n:records[n] for n in NODES})}
    return _receipt(payload)


def _check_route(route,config):
    obj=_verified(route)
    if (obj.get("schema")!=SCHEMA or obj.get("stage")!="sparse-route" or
        obj.get("seed")!=config["seed"] or
        obj.get("generation")!=config["generation"] or
        obj.get("config_sha256")!=mesh.digest(config) or
        obj.get("node_count")!=20 or obj.get("peers_per_node")!=PEERS or
        set(obj.get("routing",{}))!=set(NODES) or
        set(obj.get("probes",{}))!=set(NODES)):
        raise ValueError("routing provenance or population mismatch")
    for n in NODES:
        peers=obj["routing"][n]
        if (len(peers)!=PEERS or n in peers or
            any(p not in NODES for p in peers) or
            any(not isinstance(w,(int,float)) or w<0 or not math.isfinite(w)
                for w in peers.values()) or
            abs(sum(peers.values())-1)>1e-8):
            raise ValueError("invalid sparse five-peer probabilities")
    return obj


def second(node,config,first_receipt,route,distill=True):
    if node not in NODES: raise ValueError("unknown specialist")
    record=_verified(first_receipt)
    routed=_check_route(route,config)
    if (record.get("node")!=node or record.get("stage")!=1 or
        record.get("config_sha256")!=mesh.digest(config) or
        routed["model_sha256"][node]!=mesh.digest(record["trained_model"])):
        raise ValueError("stale or mismatched local training checkpoint")
    original=record["trained_model"]
    control=copy.deepcopy(original)
    social=copy.deepcopy(original)
    _learn_local(node,control,config["seed"],2,
                 config["update_labeled_scenes_per_node"])
    _learn_local(node,social,config["seed"],2,
                 config["update_labeled_scenes_per_node"])
    # Both branches have the same supervised examples. The only difference
    # is five-peer distillation on unlabelled, shared probe observations.
    if distill:
        peers=routed["routing"][node]
        for _epoch in range(2):
            for i in range(PROBES):
                regime=REGIMES[i%len(REGIMES)]
                track,_=observed(config["seed"]+92179,i,regime,node)
                teacher=[
                    sum(weight*routed["probes"][peer][i][axis]
                        for peer,weight in peers.items())
                    for axis in (0,1)]
                current=swarm20.predict(track,node,social)
                # Make learning conservative: retain the local forecast.
                label=[.82*current[axis]+.18*teacher[axis] for axis in (0,1)]
                _update(social,node,track,label,.003)
    payload={"schema":SCHEMA,"stage":2,"node":node,
             "seed":config["seed"],"generation":config["generation"],
             "config_sha256":mesh.digest(config),
             "route_sha256":route["sha256"],
             "first_sha256":first_receipt["sha256"],
             "control_model":control,"social_model":social,
             "local_labeled_update":config["update_labeled_scenes_per_node"],
             "unlabelled_peer_experiences":PROBES if distill else 0,
             "selected_peers":list(routed["routing"][node]),
             "communicated":bool(distill)}
    return _receipt(payload)


def _predict_all(models,seed,index,mode):
    return {node:swarm20.predict(observed(seed,index,mode,node)[0],
                                 node,models[node]) for node in NODES}


def _summarize(models,config):
    all_losses=[]
    by_regime={r:[] for r in REGIMES}
    node_losses={n:[] for n in NODES}
    seeds={}
    for seed in config["test_seeds"]:
        seed_errors=[]
        for i in range(SCENES_PER_TEST):
            mode=REGIMES[i%len(REGIMES)]
            target=target_motion(seed,i,mode)[4]
            pred=_predict_all(models,seed,i,mode)
            merged=[statistics.fmean(pred[n][axis] for n in NODES)
                    for axis in (0,1)]
            loss=mesh.squared_error(merged,target)
            seed_errors.append(loss)
            all_losses.append(loss)
            by_regime[mode].append(loss)
            for n in NODES:
                node_losses[n].append(mesh.squared_error(pred[n],target))
        seeds[str(seed)]=statistics.fmean(seed_errors)
    return {"mse":statistics.fmean(all_losses),
            "per_seed_mse":seeds,
            "per_regime_mse":{k:statistics.fmean(v) for k,v in by_regime.items()},
            "best_single_node_mse":min(statistics.fmean(v)
                                        for v in node_losses.values()),
            "best_single_node":min(NODES,key=lambda n:statistics.fmean(node_losses[n])),
            "scene_count":len(all_losses)}


def finish(config,first_folder,second_folder,route):
    first_records=_read_receipts(first_folder,1)
    second_records=_read_receipts(second_folder,2)
    routing=_check_route(route,config)
    before={}
    controls={}
    social={}
    for node in NODES:
        first_data=first_records[node]
        second_data=second_records[node]
        if (first_data.get("config_sha256")!=mesh.digest(config) or
            first_data.get("seed")!=config["seed"] or
            second_data.get("config_sha256")!=mesh.digest(config) or
            second_data.get("seed")!=config["seed"] or
            second_data.get("route_sha256")!=route["sha256"] or
            second_data.get("first_sha256")!=mesh.digest(first_data) or
            second_data.get("selected_peers")!=list(routing["routing"][node]) or
            second_data.get("communicated") is not True):
            raise ValueError("inconsistent local updates or missing peer messages")
        before[node]=first_data["trained_model"]
        controls[node]=second_data["control_model"]
        social[node]=second_data["social_model"]
        for w in (before[node],controls[node],social[node]):_check_model(w)
    baseline=_summarize(before,config)
    control=_summarize(controls,config)
    communicated=_summarize(social,config)
    report={"schema":SCHEMA,"seed":config["seed"],
            "generation":config["generation"],
            "parent_sha256":config["parent_sha256"],
            "node_count":20,"peers_per_node":PEERS,
            "physical_training_jobs_per_wave":20,
            "stage1_local_training_per_node":config["train_labeled_scenes_per_node"],
            "stage2_labeled_training_per_node":config["update_labeled_scenes_per_node"],
            "stage2_unlabelled_peer_probes_per_node":PROBES,
            "before_second_learning":baseline,
            "independent_learning_control":control,
            "sparse_peer_learning":communicated,
            "relative_change_vs_independent":communicated["mse"]/control["mse"]-1,
            "beat_independent_control":communicated["mse"]<control["mse"],
            "beat_best_individual":(
                communicated["mse"]<communicated["best_single_node_mse"]),
            "trained_models_differ":len({mesh.digest(social[n]) for n in NODES})==20,
            "candidate_only_not_promoted":True}
    checkpoint_payload={"schema":SCHEMA,"generation":config["generation"],
                        "seed":config["seed"],
                        "parent_sha256":config["parent_sha256"],
                        "nodes":social,"routing":routing["routing"]}
    return report,_receipt(checkpoint_payload)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="command",required=True)
    prep=sp.add_parser("prepare")
    prep.add_argument("--seed",type=int,required=True)
    prep.add_argument("--out",required=True)
    prep.add_argument("--parent")
    st1=sp.add_parser("first")
    st1.add_argument("--node",choices=NODES,required=True)
    st1.add_argument("--config",required=True)
    st1.add_argument("--parent")
    st1.add_argument("--out",required=True)
    agg=sp.add_parser("route")
    agg.add_argument("--config",required=True)
    agg.add_argument("--receipts",required=True)
    agg.add_argument("--out",required=True)
    st2=sp.add_parser("second")
    st2.add_argument("--node",choices=NODES,required=True)
    st2.add_argument("--config",required=True)
    st2.add_argument("--first",required=True)
    st2.add_argument("--route",required=True)
    st2.add_argument("--out",required=True)
    end=sp.add_parser("finish")
    for key in ("config","first-receipts","second-receipts","route","out"):
        end.add_argument("--"+key,required=True)
    a=p.parse_args()
    if a.command=="prepare":
        print(prepare(a.seed,a.out,a.parent))
    elif a.command=="first":
        mesh.save(a.out,first(a.node,mesh.load(a.config),a.parent))
    elif a.command=="route":
        mesh.save(a.out,aggregate_first(mesh.load(a.config),a.receipts))
    elif a.command=="second":
        mesh.save(a.out,second(a.node,mesh.load(a.config),
                               mesh.load(a.first),mesh.load(a.route)))
    else:
        r,c=finish(mesh.load(a.config),a.first_receipts,
                   a.second_receipts,mesh.load(a.route))
        mesh.save(Path(a.out)/"metrics.json",r)
        mesh.save(Path(a.out)/"checkpoint.json",c)
        print({"independent":r["independent_learning_control"]["mse"],
               "sparse":r["sparse_peer_learning"]["mse"],
               "beats_control":r["beat_independent_control"],
               "candidate_sha256":c["sha256"]})


if __name__=="__main__":
    main()
