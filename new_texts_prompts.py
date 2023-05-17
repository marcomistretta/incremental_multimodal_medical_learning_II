import random

CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}


def generate_chexpert_class_prompts(train_logit_diff=False, n=4):
    """Generate text prompts for each CheXpert classification task
    Parameters
    ----------
    n:  int
        number of prompts per class
    Returns
    -------
    class prompts : dict
        dictionary of class to prompts
    """
    if n != 1 and n != 4:
        raise ValueError("n must be 1 or 4")
    ONLY_POS = not train_logit_diff
    print("/// medclip prompts ///")
    if ONLY_POS:
        print("/// only pos ///")
    else:
        print("/// pos and neg ///")
    print("/// number of pos:", n, " ///")
    OPZ = 1
    prompts = {}
    for k, v in CHEXPERT_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        if ONLY_POS:
            if n == 1:
                prompts[k] = {
                    "positive": random.sample(cls_prompts, n),
                    "negative": [f"No evidence of {k}"],
                }

            elif n == 4:
                prompts[k] = {
                    "positive": random.sample(cls_prompts, n),
                    "negative": [f"There is no {c}", f"No evidence of {c}",
                                 f"No evidence of acute {c}", f"No signs of {c}"],
                }
        else:
            prompts[k] = random.sample(cls_prompts, n)

    if not ONLY_POS:
        diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        pos_neg_prompts = {}

        if OPZ == 1:
            print("/// 4 neg ///")
            pos_neg_prompts[diseases[0]] = {
                "positive": prompts["Atelectasis"],
                "negative": [f"There is no Atelectasis", "No evidence of Atelectasis",
                             "No evidence of acute Atelectasis", "No signs of Atelectasis"]
            }
            pos_neg_prompts[diseases[1]] = {
                "positive": prompts["Cardiomegaly"],
                "negative": ["There is no Cardiomegaly", "No evidence of Cardiomegaly",
                             "No evidence of acute Cardiomegaly", "No signs of Cardiomegaly"]
            }
            pos_neg_prompts[diseases[2]] = {
                "positive": prompts["Consolidation"],
                "negative": ["There is no Consolidation", "No evidence of Consolidation",
                             "No evidence of acute Consolidation", "No signs of Consolidation"]
            }
            pos_neg_prompts[diseases[3]] = {
                "positive": prompts["Edema"],
                "negative": ["There is no Edema", "No evidence of Edema",
                             "No evidence of acute Edema", "No signs of Edema"]
            }
            pos_neg_prompts[diseases[4]] = {
                "positive": prompts["Pleural Effusion"],
                "negative": ["There is no Pleural Effusion", "No evidence of Pleural Effusion",
                             "No evidence of acute Pleural Effusion", "No signs of Pleural Effusion"]
            }
        elif OPZ == 2:
            print("/// only one neg ///")
            pos_neg_prompts[diseases[0]] = {
                "positive": prompts["Atelectasis"],
                "negative": ["There is no findings"]
            }
            pos_neg_prompts[diseases[1]] = {
                "positive": prompts["Cardiomegaly"],
                "negative": ["There is no findings"]
            }
            pos_neg_prompts[diseases[2]] = {
                "positive": prompts["Consolidation"],
                "negative": ["There is no findings"]
            }
            pos_neg_prompts[diseases[3]] = {
                "positive": prompts["Edema"],
                "negative": ["There is no findings"]
            }
            pos_neg_prompts[diseases[4]] = {
                "positive": prompts["Pleural Effusion"],
                "negative": ["There is no findings"]
            }
    if ONLY_POS:
        return prompts
    else:
        return pos_neg_prompts
