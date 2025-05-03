def trigger_em(pred, gold):
    return int(pred["trigger"] == gold["trigger"])

def bag_of_steps(pred, gold):
    p = {s["name"] for s in pred["steps"]}
    g = {s["name"] for s in gold["steps"]}
    return len(p & g) / len(g)

def halluc_step(pred, gold):
    p = {s["name"] for s in pred["steps"]}
    g = {s["name"] for s in gold["steps"]}
    return len(p - g) / max(1, len(p))

def halluc_table(pred, gold):
    return int(pred["trigger"]["table"] != gold["trigger"]["table"])
