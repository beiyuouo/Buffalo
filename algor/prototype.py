import torch
import numpy as np
from loguru import logger
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
# from pyclustering.cluster.kmeans import kmeans


def weighted_distance(X, Y):
    mean_x, std_x, weight_x = X
    mean_y, std_y, weight_y = Y
    return (np.power((mean_x - mean_y), 2).sum() + np.power((std_x - std_y), 2).sum()) * (1 + weight_x)

def cluster_centroids(X, labels, k):
    centroids = []
    for i in range(k):
        cluster = [X[j] for j in range(len(X)) if labels[j] == i]

        if len(cluster) == 0:
            centroids.append([np.zeros_like(X[0][0]), 0, 0])
            continue

        cmean = [cluster[j][0] for j in range(len(cluster))]
        cstd = [cluster[j][1] for j in range(len(cluster))]
        cweight = [cluster[j][2] for j in range(len(cluster))]

        # quick fix
        if np.sum(cweight) == 0:
            cweight = np.ones(len(cweight)) / len(cweight)

        # logger.debug(f"cmean: {cmean}")
        # logger.debug(f"cstd: {cstd}")
        # logger.debug(f"cweight: {cweight}")

        mean = np.average(cmean, axis=0, weights=cweight)
        std = np.average(cstd, axis=0, weights=cweight)
        weight = np.sum(cweight)

        # logger.debug(f"mean: {mean}, std: {std}, weight: {weight}")
        centroids.append([mean, std, weight])
    
    # normalize weights
    weights = [centroids[j][2] for j in range(k)]
    weights = np.array(weights) / np.sum(weights)
    for j in range(k):
        centroids[j][2] = weights[j]

    return centroids

def wkmeans(X, k, centroids=None, steps=100):
    # FIXME: check again
    k = min(k, len(X))
    if centroids is None:
        centroids = [X[i] for i in np.random.choice(len(X), k, replace=False)]
    for _ in range(steps):
        dist = np.array([[weighted_distance(X[i], c) for i in range(len(X))] for c in centroids])
        # logger.debug(f"dist.shape: {dist.shape}")
        labels = dist.argmin(axis=0)
        new_centroids = cluster_centroids(X, labels, k)
        # check list equality
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

def wfpa(cfg, prototypes, round_idx, client_ids=None):
    cluster_weights_i = []
    cluster_weights_t = []
    mus_i = []
    mus_t = []
    sigmas_i = []
    sigmas_t = []
    cross_matrixs = []
    X_img = []
    X_txt = []

    for prototype in prototypes:
        (img_protos, text_protos, cross_matrix, proto_cnt_i, proto_cnt_t, proto_std_i, proto_std_t) = prototype
        mus_i.extend(img_protos)
        mus_t.extend(text_protos)
        sigmas_i.extend(proto_std_i)
        sigmas_t.extend(proto_std_t)

        cluster_weights_i.extend(proto_cnt_i)
        cluster_weights_t.extend(proto_cnt_t)

        cross_matrixs.append(cross_matrix)
    
    # normalize weights
    cluster_weights_i = np.array(cluster_weights_i) / np.sum(cluster_weights_i)
    cluster_weights_t = np.array(cluster_weights_t) / np.sum(cluster_weights_t)

    for mean_i, std_i, weight_i in zip(mus_i, sigmas_i, cluster_weights_i):
        X_img.append([mean_i, std_i, weight_i])

    for mean_t, std_t, weight_t in zip(mus_t, sigmas_t, cluster_weights_t):
        X_txt.append([mean_t, std_t, weight_t])

    # X_img = np.array(X_img)
    # X_txt = np.array(X_txt)

    # logger.debug(f"X_img.shape: {X_img.shape}")
    # logger.debug(f"X_txt.shape: {X_txt.shape}")

    # KMeans
    kmeans_img, label_img = wkmeans(X_img, cfg.train.cross_fedproto.k, steps=100)
    kmeans_txt, label_txt = wkmeans(X_txt, cfg.train.cross_fedproto.k, steps=100)

    logger.debug(f"KMeans cluster centroids: {kmeans_img}, {kmeans_txt}")
    logger.debug(f"KMeans cluster labels: {label_img}, {label_txt}")

    new_cross_matrix = np.zeros((cfg.train.cross_fedproto.k, cfg.train.cross_fedproto.k))

    for idx, cross_matrix in enumerate(cross_matrixs):
        id = idx * cfg.train.cross_fedproto.k
        for i in range(cfg.train.cross_fedproto.k):
            for j in range(cfg.train.cross_fedproto.k):
                new_cross_matrix[label_img[id + i]][label_txt[id + j]] += cross_matrix[i][j]
    
    return kmeans_img, kmeans_txt, new_cross_matrix

def get_img_proto_props(img_embs, img_proto):
    # TODO
    img_embs = img_embs.reshape(-1, img_embs.shape[-1]).cpu().detach().numpy()
    props = np.zeros((img_embs.shape[0], len(img_proto)))
    for i in range(img_embs.shape[0]):
        for k in range(len(img_proto)):
            mean, std, weight = img_proto[k]
            if std == 0:
                prop = 0
            else:
                prop = np.exp(-((img_embs[i] - mean) ** 2).sum() / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
            props[i][k] = prop * weight
            
    # normalize with softmax
    props = np.exp(props) / np.sum(np.exp(props), axis=1, keepdims=True)
    prop_idx = np.argmax(props, axis=1)
    return props, prop_idx

def get_txt_proto_props(txt_embs, txt_proto):
    # TODO
    txt_embs = txt_embs.reshape(-1, txt_embs.shape[-1]).cpu().detach().numpy()
    props = np.zeros((txt_embs.shape[0], len(txt_proto)))
    for i in range(txt_embs.shape[0]):
        for k in range(len(txt_proto)):
            mean, std, weight = txt_proto[k]
            if std == 0:
                prop = 0
            else:
                prop = np.exp(-((txt_embs[i] - mean) ** 2).sum() / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
            props[i][k] = prop * weight
    
    # normalize with softmax
    props = np.exp(props) / np.sum(np.exp(props), axis=1, keepdims=True)
    prop_idx = np.argmax(props, axis=1)
    return props, prop_idx

def get_miss_loss(img_embs, img_protos):
    batch_size, seq_len, emb_dim = img_embs.shape
    img_embs = img_embs.reshape(-1, img_embs.shape[-1])
    miss_loss = 0
    total_dist = 0
    for i in range(img_embs.shape[0]):
        min_dist = np.inf
        min_idx = -1
        for idx, (mean, std, weight) in enumerate(img_protos):
            mean = torch.tensor(mean).to(img_embs.device)
            std = torch.tensor(std).to(img_embs.device)
            weight = torch.tensor(weight).to(img_embs.device)
            dist = ((img_embs[i] - mean) ** 2).sum() * weight / img_embs.shape[1]
            if dist < min_dist:
                min_dist = dist
                min_idx = idx

        total_dist = 0
        for idx, (mean, std, weight) in enumerate(img_protos):
            mean = torch.tensor(mean).to(img_embs.device)
            std = torch.tensor(std).to(img_embs.device)
            weight = torch.tensor(weight).to(img_embs.device)
            dist = ((img_embs[i] - mean) ** 2).sum() * weight / img_embs.shape[1]
            if idx != min_idx:
                total_dist += dist
        miss_loss += min_dist

    return miss_loss / img_embs.shape[0] - total_dist / img_embs.shape[0] / len(img_protos)


def get_sim_loss(img_embs, proto_embs, img_proto_ids, img_protos, txt_embs, txt_proto_ids, txt_protos):
    sim_loss = 0.0
    img_sim = torch.zeros_like(img_embs[0][0]).to(img_embs.device)
    txt_sim = torch.zeros_like(txt_embs[0][0]).to(txt_embs.device)
    img_embs = img_embs.reshape(-1, img_embs.shape[-1])
    img_proto_ids = img_proto_ids.reshape(-1)
    proto_embs = proto_embs.reshape(-1, proto_embs.shape[-1])
    txt_embs = txt_embs.reshape(-1, txt_embs.shape[-1])
    txt_proto_ids = txt_proto_ids.reshape(-1)

    for i in range(img_embs.shape[0]):
        img_proto = proto_embs[i]
        # convert dtype as same as img_embs
        # img_proto = img_proto.to(torch.float16)
        img_sim += ((img_embs[i] - img_proto) ** 2).sum() / img_embs.shape[1] * torch.tensor(img_protos[img_proto_ids[i]][1])
        # img_sim += torch.dot(img_embs[i], img_proto) / (torch.norm(img_embs[i]) * torch.norm(img_proto))

    for i in range(txt_embs.shape[0]):
        txt_proto = torch.tensor(txt_protos[txt_proto_ids[i]][0]).to(txt_embs.device)
        txt_sim += ((txt_embs[i] - txt_proto) ** 2).sum() / txt_embs.shape[1] * torch.tensor(txt_protos[txt_proto_ids[i]][1])
        # with torch.autocast(device_type="cuda"):
        # txt_sim += torch.dot(txt_embs[i], txt_proto) / (torch.norm(txt_embs[i]) * torch.norm(txt_proto))

    # print(img_sim)
    # print(txt_sim)

    img_sim = img_sim / img_embs.shape[0]
    txt_sim = txt_sim / txt_embs.shape[0]

    sim_loss = (img_sim - txt_sim) ** 2
    sim_loss = sim_loss.mean()

    return sim_loss


if __name__ == "__main__":
    import ezkfg as ez
    cfg = ez.load("config/mrg_config.yaml")
    cfg.train.cross_fedproto.k = 1
    prototypes = [
        (np.array([[1, 1], [2, 2]]), np.array([[3, 3], [4, 4]]), np.array([[1, 2], [3, 4]]), [0.5, 0.5], [0.5, 0.5], [0.1, 0.1], [0.1, 0.1]),
        (np.array([[5, 5], [6, 6]]), np.array([[7, 7], [8, 8]]), np.array([[5, 6], [7, 8]]), [0.5, 0.5], [0.5, 0.5], [0.1, 0.1], [0.1, 0.1])
    ]
    _ = wfpa(cfg, prototypes, 0, [0, 1])
    print(_)

    img_embs = torch.randn(2, 2, 2).to("cuda")
    img_protos = [[[1, 1], 1, 0.5], [[2, 2], 2, 0.5]]
    print(get_miss_loss(img_embs, img_protos))

    img_embs = torch.randn(2, 2, 2).to("cuda")
    img_proto_ids = np.array([0, 1, 0, 1])
    img_protos = [[[1, 1], 0.5], [[2, 2], 0.5]]
    txt_embs = torch.randn(2, 2, 2).to("cuda")
    txt_proto_ids = np.array([0, 1, 0, 1])
    txt_protos = [[[1, 1], 0.5], [[2, 2], 0.5]]
    print(get_sim_loss(img_embs, img_proto_ids, img_protos, txt_embs, txt_proto_ids, txt_protos))

    print("Done!")
