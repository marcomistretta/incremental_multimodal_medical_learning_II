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


def generate_chexpert_class_prompts(n: int = 5):
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

        prompts[k] = random.sample(cls_prompts, n)

    # diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    # pos_neg_prompts = {}
    # pos_neg_prompts[diseases[0]] = {
    #     "positive": prompts["Atelectasis"],
    #     "negative": ["No signs of collapsed lung tissue were found on the X-ray",
    #                  "The patient's lungs appear clear and functioning properly",
    #                  "There is no evidence of atelectasis on the CT scan",
    #                  "The X-ray did not reveal any abnormalities associated with atelectasis",
    #                  "The doctor confirmed that there is no atelectasis present in the patient's lungs"]
    # }
    #
    # pos_neg_prompts[diseases[1]] = {
    #     "positive": prompts["Cardiomegaly"],
    #     "negative": ["The patient's heart appears normal in size and function",
    #                  "The echocardiogram showed no signs of cardiomegaly",
    #                  "The X-ray did not reveal any enlargement of the heart",
    #                  "The doctor confirmed that there is no evidence of cardiomegaly in the patient's heart",
    #                  "There are no symptoms or indications of cardiomegaly present in the patient"]
    # }
    #
    # pos_neg_prompts[diseases[2]] = {
    #     "positive": prompts["Consolidation"],
    #     "negative": ["The X-ray and CT scan showed no evidence of lung consolidation",
    #                  "There are no signs of consolidation on the patient's chest exam",
    #                  "The doctor confirmed that there is no lung consolidation present in the patient",
    #                  "The patient's lungs appear clear and healthy with no signs of consolidation",
    #                  "The radiologist found no evidence of consolidation in the lung tissue"]
    # }
    #
    # pos_neg_prompts[diseases[3]] = {
    #     "positive": prompts["Edema"],
    #     "negative": ["The patient's legs and feet show no signs of swelling or edema",
    #                  "The doctor confirmed that there is no edema present in the patient's body",
    #                  "The ultrasound did not reveal any signs of edema in the patient's tissues",
    #                  "The patient has no symptoms or indications of edema",
    #                  "The medical tests showed no evidence of fluid accumulation or edema"]
    # }
    #
    # pos_neg_prompts[diseases[4]] = {
    #     "positive": prompts["Pleural Effusion"],
    #     "negative": ["There is no sign of fluid accumulation in the pleural space",
    #                  "The chest X-ray showed no evidence of pleural effusion",
    #                  "The patient's breathing sounds clear and there are no signs of pleural effusion",
    #                  "The doctor confirmed that there is no pleural effusion present in the patient",
    #                  "The ultrasound did not detect any pleural effusion in the patient's chest"]
    # }
    # return pos_neg_prompts
    return prompts
