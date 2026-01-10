import keras
import os
from tqdm import tqdm
import numpy as np

import logging
logger = logging.getLogger(__name__)


def load_model(path_to_model):
    logger.info(f"Loading keras model at {path_to_model}")
    model = keras.models.load_model(path_to_model)
    return model


def run_house_nohouse_inference(df5, img_file_dir, model, confidence_thresh:float=0.65):
    logger.info("Running house/no-house inference")
    df5 = df5.copy()
    house_nohouse_model = load_model(model)
    is_house = []
    for idx, row in tqdm(df5.iterrows(), total=len(df5)):
        building_id = row['building_id']
        image_wall_id = row["ID"]
        out_name_rect = f"{building_id}_{image_wall_id}_rectified.jpg"

        img_path = os.path.join(img_file_dir, out_name_rect)
        if not os.path.exists(img_path):
            is_house.append(None)
            continue
        try:
            img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = keras.applications.mobilenet.preprocess_input(img_array)
            img_batch = np.expand_dims(img_array, axis=0)
            predictions = house_nohouse_model.predict(img_batch)
            score = predictions[0][0]
            is_house.append(score > confidence_thresh)
        except Exception as e:
            logger.debug(f"Could not run inferene on image {out_name_rect}, because of error: {e}")
            is_house.append(None)
            continue

    df5['is_house'] = is_house
    return df5
