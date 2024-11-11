import os
import sys
import ezkfg as ez
from copy import deepcopy
from loguru import logger
from main_fed import main

if __name__ == "__main__":
    # check niid in sys.argv
    niid = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "niid":
            niid = True

    print("niid:", niid)

    cfg_file = os.path.join(os.path.dirname(__file__), "config", "mrg_config.yaml")
    # cfg_file = os.path.join(os.path.dirname(__file__), "config", "mvqa_config.yaml")
    cfg = ez.load(cfg_file)

    if not niid:
        cfg.niid = False
        cfg.train.modal_hetero = False
        logger.info("Attention: Modal Homogeneous")
        main(deepcopy(cfg))

        cfg.train.modal_hetero = True
        for mm_strategy in ["prototype", "mean", "drop", "duplicate"]:
            cfg["train"]["mm_strategy"] = mm_strategy
            if mm_strategy == "mean":
                for mean_strategy in ["token", "original"]:
                    cfg["train"]["mm_mean_type"] = mean_strategy
                    print("Modal Heterogeneous, mm_strategy:", mm_strategy, ", mean_strategy:", mean_strategy)
                    logger.info("Attention: Modal Heterogeneous, mm_strategy: {}, mean_strategy: {}".format(mm_strategy, mean_strategy))
                    main(deepcopy(cfg))
            else:
                print("Modal Heterogeneous, mm_strategy:", mm_strategy)
                logger.info("Attention: Modal Heterogeneous, mm_strategy: {}".format(mm_strategy))
                main(deepcopy(cfg))

    else:
        cfg.niid = True

        for niid_type in ["feature", "quantity"]:
            cfg["data"]["niid_type"] = niid_type

            cfg.train.modal_hetero = False
            print("Modal Homogeneous, niid_type:", niid_type)
            logger.info("Attention: Modal Homogeneous, niid_type: {}".format(niid_type))
            main(deepcopy(cfg))

            cfg.train.modal_hetero = True
            for mm_strategy in ["prototype", "mean", "drop", "duplicate"]:
                cfg["train"]["mm_strategy"] = mm_strategy
                if mm_strategy == "mean":
                    for mean_strategy in ["token", "original"]:
                        cfg["train"]["mm_mean_type"] = mean_strategy
                        print("Modal Heterogeneous, niid_type:", niid_type, ", mm_strategy:", mm_strategy, ", mean_strategy:", mean_strategy)
                        logger.info("Attention: Modal Heterogeneous, niid_type: {}, mm_strategy: {}, mean_strategy: {}".format(niid_type, mm_strategy, mean_strategy))
                        main(deepcopy(cfg))
                else:
                    print("Modal Heterogeneous, niid_type:", niid_type, ", mm_strategy:", mm_strategy)
                    logger.info("Attention: Modal Heterogeneous, niid_type: {}, mm_strategy: {}".format(niid_type, mm_strategy))
                    main(deepcopy(cfg))

    # cfg = ez.load(cfg_file)

    # for lr in [0.0003, 0.0001, 0.00006]:
    #     cfg["train"]["lr"] = lr
    #     cfg["train"]["lr_ve"] = lr
    #     cfg["train"]["lr_proj"] = lr / 3
    #     main(cfg)

    # cfg = ez.load(cfg_file)

    # for weight in [0.1, 0.01, 0.001]:
    #     cfg["train"]["loss"]["weight_proto"] = weight
    #     cfg["train"]["loss"]["weight_sim"] = weight
    #     main(cfg)
