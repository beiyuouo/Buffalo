import torch
from loguru import logger

def fedavg(cfg, client_params, round_idx, **kwargs):
    """Federated Averaging algorithm.

    Args:
        cfg (dict): Configuration dictionary.
        client_params (list): List of client parameters.
        round_idx (int): Current round index.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: Updated global model parameters.
    """
    # Initialize global model parameters
    global_params = {k: torch.zeros_like(v) for k, v in client_params[0].items()}

    if cfg.train.modal_hetero and cfg.train.mm_strategy == "prototype":
        logger.info(f"Use strategy {cfg.train.mm_strategy}")
        client_ids = kwargs["client_ids"]
        client_types = kwargs["client_types"]
        ve_weights = []
        pj_weights = []

        for client_id in client_ids:
            client_type = client_types[client_id]
            if client_type == 0:
                ve_weights.append(1)
                pj_weights.append(1)
            else:
                ve_weights.append(1/3)
                pj_weights.append(1/3)

        ve_weights = torch.tensor(ve_weights)
        ve_weights = ve_weights / ve_weights.sum()
        pj_weights = torch.tensor(pj_weights)
        pj_weights = pj_weights / pj_weights.sum()

        logger.debug(f"client_types: {client_types}")
        logger.debug(f"client_ids: {client_ids}")
        logger.debug(f"ve_weights: {ve_weights}")
        logger.debug(f"pj_weights: {pj_weights}")
        

        for id, client_param in enumerate(client_params):
            for k, v in client_param.items():
                if "visual_encoder" in k:
                    global_params[k] += ve_weights[id] * v
                else:
                    global_params[k] += pj_weights[id] * v

    else:
        # Aggregate client parameters
        for client_param in client_params:
            for k, v in client_param.items():
                global_params[k] += v

        # Average client parameters
        for k in global_params.keys():
            global_params[k] /= len(client_params)

    logger.debug(f"Round {round_idx}: Global model parameters: {global_params.keys()}")

    return global_params
