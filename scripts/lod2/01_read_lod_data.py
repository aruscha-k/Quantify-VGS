#!/usr/bin/env python
"""
LOD2 GML → cleaned building parts / walls / roofs pipeline.

Steps:
1. Read all .gml files in an input directory.
2. For each file, parse buildings and parts:
   - parts dataframe (df1)
   - walls dataframe (df2)
   - roofs dataframe (df3)
3. Clean coordinate lists (remove duplicates, ensure closed rings).
4. Detect and remove duplicate walls where:
   - the corresponding parts have identical ground surfaces, and
   - the wall surface coordinates are identical.
5. Save final dataframes as pickles in an output directory.
"""

import logging
from datetime import datetime
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

import geopandas as gpd
import pandas as pd
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon

import scripts.config as CONF
# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
# ------
# logger
# ------
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
logging.basicConfig(
    level=logging.DEBUG,                                   # INFO/ DEBUG
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f"{CONF.log_file_dir}/{timestamp}_read_lod_data.log"),    # log to file
        logging.StreamHandler()                             # log to console
    ]
)


# ---------------------------------------------------------------------
# Helpers for parsing XML / geometry
# ---------------------------------------------------------------------

def convert_coordinate_pos_list_to_tuple(coords_as_text: str) -> List[Tuple[float, float, float]]: #unchanged
    """
    Converts a string of coordinates into a list of (x, y, z) tuples.
    Assumes coords_as_text is a whitespace-separated sequence of numbers.
    """
    coords_list = [float(coord) for coord in coords_as_text.split()]
    # Group as triples
    tuple_list = [
        (coords_list[i], coords_list[i + 1], coords_list[i + 2])
        for i in range(0, len(coords_list), 3)
    ]
    return tuple_list


# taken from existing code
def extract_building_address(building, nspace):
    ''' 
    For a single building, extract adress and convert from one string to 3 values: street, house_nr and hnr_add (addition to house nr)
    '''
    try:
        for item in building.find('bldg:address', nspace).iter():
            if item.tag == "{urn:oasis:names:tc:ciq:xsdschema:xAL:2.0}LocalityName":
                district = item.text
            if item.tag == "{urn:oasis:names:tc:ciq:xsdschema:xAL:2.0}ThoroughfareName":
                street = item.text
            if item.tag == "{urn:oasis:names:tc:ciq:xsdschema:xAL:2.0}ThoroughfareNumber":
                house_nr = item.text
        try:
            house_nr = int(house_nr)
            hnr_add = ""
        except ValueError:
            for idx, char in enumerate(house_nr):
                if not char.isdigit():
                    house_nr = house_nr[:idx]
                    hnr_add = char
    except AttributeError:
        logging.debug("No address information found for building id %s", building.attrib.get("{http://www.opengis.net/gml}id", "unknown"))
        district, street, house_nr, hnr_add = None, None, None, None

    return district, street, house_nr, hnr_add


def extract_building_properties(
    building: ET.Element,
    is_part: bool,
    ns: Dict[str, str],
) -> Tuple[str, str, List[Dict[str, Any]], float, List[Tuple[float, float, float]],
           List[Dict[str, Any]], str, str, str]:
    """
    For one building or building part, extract:
    - building_id
    - roof_form
    - roof_surfaces: list of dicts {building_id, surface_id, surface_coordinates}
    - building_height
    - ground_surface coordinates (list of tuples)
    - wall_surfaces: list of dicts {building_id, wall_id, surface_coordinates}
    - street, house_nr, hnr_add
    """
    # ID
    building_id = building.attrib["{http://www.opengis.net/gml}id"] 

    ##num stories
    try: 
        num_storeys = int(building.find('bldg:storeysAboveGround', ns).text)
    except AttributeError as e:
        num_storeys = None

    ### BUILDING HEIGHT
    try:    
        building_height = float(building.find("bldg:measuredHeight", ns).text)
    except AttributeError:
        building_height = 0.0

    ### ROOF FORM
    try:
        roof_form = int(building.find("bldg:roofType", ns).text)
    except AttributeError:
        roof_form = ""

    # Wall surfaces
    wall_surfaces: List[Dict[str, Any]] = []
    wall_surface_items = building.findall("bldg:boundedBy/bldg:WallSurface", ns)
    if wall_surface_items != []:
        for wall_item in wall_surface_items:
            try:
                # get wall_id and coordinates of surface
                wall_surface_coords = convert_coordinate_pos_list_to_tuple(wall_item.find('.//gml:posList', ns).text)
                wall_surface_id = wall_item.attrib['{http://www.opengis.net/gml}id']
                wall_surfaces.append({'building_id': building_id, 'wall_id': wall_surface_id, 'surface_coordinates': wall_surface_coords})
            except KeyError:
                logging.debug("Missing wall_id or coordinates for building id %s", building_id)
                wall_surfaces.append({'building_id': '', 'wall_id': '', 'surface_coordinates': []})

    # Roof surfaces
    roof_surfaces: List[Dict[str, Any]] = []
    roof_surface_items = building.findall("bldg:boundedBy/bldg:RoofSurface", ns)
    if roof_surface_items != []:
        for roof_item in roof_surface_items:
            try:
                # get surface_id and coordinates of surface
                roof_surface_coords = convert_coordinate_pos_list_to_tuple(roof_item.find('.//gml:posList', ns).text)
                roof_surface_id = roof_item.attrib['{http://www.opengis.net/gml}id']
                roof_surfaces.append({'building_id': building_id, 'surface_id': roof_surface_id, 'surface_coordinates': roof_surface_coords})
            except KeyError:
                logging.debug("Missing roof_id or coordinates for building id %s", building_id)
                roof_surfaces.append({'building_id': '', 'surface_id': '', 'surface_coordinates': []})


    # Ground surface
    ground_surface_coords: List[Tuple[float, float, float]] = []
    ground_surface = building.find("bldg:boundedBy/bldg:GroundSurface", ns)
    if ground_surface is not None:
        pos_node = ground_surface.find(".//gml:posList", ns)
        if pos_node is not None and pos_node.text is not None:
            ground_surface_coords = convert_coordinate_pos_list_to_tuple(pos_node.text)
    else:
        logging.debug("No ground surface found for building id %s", building_id)

    # Address
    district, street, house_nr, hnr_add = extract_building_address(building, ns)

    return (
        building_id,
        roof_form,
        roof_surfaces,
        building_height,
        num_storeys,
        ground_surface_coords,
        wall_surfaces,
        street,
        house_nr,
        hnr_add,
        district
    )


# ---------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------

def read_lod2_file(lod2_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse a single GML file and extract three dataframes:
    - df1_parts: per building/part attributes
    - df2_walls: per wall
    - df3_roofs: per roof surface
    """
    ns = {
        "gml": "http://www.opengis.net/gml",
        "core": "http://www.opengis.net/citygml/1.0",
        "bldg": "http://www.opengis.net/citygml/building/1.0",
        "gen": "http://www.opengis.net/citygml/generics/1.0",
    }

    xml_file = ET.parse(str(lod2_file))
    root = xml_file.getroot()

    all_city_objs = root.findall("core:cityObjectMember", ns)
    logging.info("File %s: %d city objects", lod2_file.name, len(all_city_objs))

    df1_parts_list: List[Dict[str, Any]] = []
    df2_walls_list: List[Dict[str, Any]] = []
    df3_roofs_list: List[Dict[str, Any]] = []

    for obj in all_city_objs:
        building = obj.find("bldg:Building", ns)
        if building is None:
            continue

        # Simple check if there is geometry at all
        if not building.findall(".//bldg:boundedBy", ns):
            continue

        building_id = building.attrib["{http://www.opengis.net/gml}id"]
        building_parts = building.findall("bldg:consistsOfBuildingPart", ns)
        has_building_parts = len(building_parts) > 0

        # Main building
        (
            building_id,
            roof_form,
            roof_surfaces,
            building_height,
            num_storeys,
            ground_surface_coords,
            wall_surfaces,
            street,
            house_nr,
            hnr_add,
            district
        ) = extract_building_properties(building, is_part=False, ns=ns)

        df1_parts_list.append(
            {
                "building_id": building_id,
                "has_building_parts": has_building_parts,
                "is_part": False,
                "street": street,
                "house_nr": house_nr,
                "hnr_add": hnr_add,
                "num_storeys": num_storeys,
                "district": district,
                "roof_form": roof_form,
                "building_height": building_height,
                "ground_surface": ground_surface_coords,
            }
        )
        df2_walls_list.extend(wall_surfaces)
        df3_roofs_list.extend(roof_surfaces)

        # Building parts
        if has_building_parts:
            for bp in building_parts:
                building_part = bp.find("bldg:BuildingPart", ns)
                if building_part is None:
                    continue
                (
                    building_id,
                    roof_form,
                    roof_surfaces,
                    building_height,
                    num_storeys,
                    ground_surface_coords,
                    wall_surfaces,
                    street,
                    house_nr,
                    hnr_add,
                    district
                ) = extract_building_properties(building_part, is_part=True, ns=ns)

                df1_parts_list.append(
                    {
                        "building_id": building_id,
                        "has_building_parts": has_building_parts,
                        "is_part": True,
                        "street": street,
                        "house_nr": house_nr,
                        "hnr_add": hnr_add,
                        "num_storeys": num_storeys,
                        "district": district,
                        "roof_form": roof_form,
                        "building_height": building_height,
                        "ground_surface": ground_surface_coords,
                    }
                )
                df2_walls_list.extend(wall_surfaces)
                df3_roofs_list.extend(roof_surfaces)

    df1_parts = pd.DataFrame(df1_parts_list)
    df2_walls = pd.DataFrame(df2_walls_list)
    df3_roofs = pd.DataFrame(df3_roofs_list)

    return df1_parts, df2_walls, df3_roofs


def create_gdf_for_all(df1_parts, df2_walls, df3_roofs, output_file):
    """
    Aggregate all extracted LOD2 information on building_id level
    and save as pickle.
    """

    # --- Aggregate walls per building ---
    if not df2_walls.empty:
        walls_agg = (
            df2_walls.groupby("building_id", as_index=False)
            .agg(list)
            .rename(columns={"wall_id": "wall_id_list", "surface_coordinates": "wall_coords_list"})
        )
        # Convert each row to dict format for consistency
        walls_agg["wall_surfaces"] = walls_agg.apply(
            lambda row: [
                {"building_id": row.building_id, "wall_id": wid, "surface_coordinates": coords}
                for wid, coords in zip(row["wall_id_list"], row["wall_coords_list"])
            ],
            axis=1
        )
        walls_agg = walls_agg[["building_id", "wall_surfaces"]]
    else:
        walls_agg = pd.DataFrame(columns=["building_id", "wall_surfaces"])

    # --- Aggregate roofs per building ---
    if not df3_roofs.empty:
        roofs_agg = (
            df3_roofs.groupby("building_id", as_index=False)
            .agg(list)
            .rename(columns={"surface_id": "roof_id_list", "surface_coordinates": "roof_coords_list"})
        )
        roofs_agg["roof_surfaces"] = roofs_agg.apply(
            lambda row: [
                {"building_id": row.building_id, "surface_id": rid, "surface_coordinates": coords}
                for rid, coords in zip(row["roof_id_list"], row["roof_coords_list"])
            ],
            axis=1
        )
        roofs_agg = roofs_agg[["building_id", "roof_surfaces"]]
    else:
        roofs_agg = pd.DataFrame(columns=["building_id", "roof_surfaces"])

    # --- Base building dataframe ---
    buildings = df1_parts.drop_duplicates(subset="building_id").reset_index(drop=True)

    # --- Merge walls and roofs ---
    all_building_df = buildings.merge(walls_agg, on="building_id", how="left")
    all_building_df = all_building_df.merge(roofs_agg, on="building_id", how="left")

    # --- Fill missing wall/roof lists ---
    all_building_df["wall_surfaces"] = all_building_df["wall_surfaces"].apply(lambda x: x if isinstance(x, list) else [])
    all_building_df["roof_surfaces"] = all_building_df["roof_surfaces"].apply(lambda x: x if isinstance(x, list) else [])

    print(f"Total buildings aggregated: {len(all_building_df)}")

    # --- Save pickle ---
    output_path = Path(output_file)
    with open(output_path, "wb") as f:
        pickle.dump(all_building_df, f)

    return all_building_df
# ---------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------

def remove_double_coordinates(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Remove duplicate coordinates from lists of coordinate tuples in df[col_name].
    Ensures that polygons are closed rings (first == last) if there is at least one coordinate.
    """
    changed = 0

    for idx, row in df.iterrows():
        coords = row.get(col_name)
        if not coords:
            continue

        # Deduplicate while preserving order
        seen = set()
        new_coords = []
        for c in coords:
            if c not in seen:
                new_coords.append(c)
                seen.add(c)

        # Close polygon if needed
        if len(new_coords) > 2 and new_coords[0] != new_coords[-1]:
            new_coords.append(new_coords[0])

        if new_coords != coords:
            df.at[idx, col_name] = new_coords
            changed += 1

    logging.info("%d coordinate lists changed in column '%s'", changed, col_name)
    return df


def get_duplicate_wall_ids_for_two_parts(df2_walls_all: pd.DataFrame, building_id1: str, building_id2: str) -> List[str]:
    """
    For two parts with the same ground surface, find wall surfaces that are identical.
    Returns wall_ids from part2 to be removed.
    """
    part1_walls = df2_walls_all[df2_walls_all["building_id"] == building_id1]
    part2_walls = df2_walls_all[df2_walls_all["building_id"] == building_id2]

    # Map surface_coordinates of part2 to wall_id
    surface_to_wall2 = {}
    for _, w2 in part2_walls.iterrows():
        coords = w2["surface_coordinates"]
        if not coords:
            continue
        key = tuple(coords)
        surface_to_wall2[key] = w2["wall_id"]

    ids_to_remove = []
    for _, w1 in part1_walls.iterrows():
        coords1 = w1["surface_coordinates"]
        if not coords1:
            continue
        key1 = tuple(coords1)
        if key1 in surface_to_wall2:
            # Remove wall from part2
            ids_to_remove.append(surface_to_wall2[key1])

    return ids_to_remove


def find_wall_ids_to_remove(
    df1_parts_all: pd.DataFrame, df2_walls_all: pd.DataFrame
) -> Tuple[Set[str], int]:
    """
    Identify duplicate walls across parts:
    - Parts that share exactly the same ground surface geometry.
    - Walls whose surface_coordinates match.

    Returns:
    - set of wall_ids to remove
    - count of part-pairs that shared the same ground surface
    """
    if df1_parts_all.empty or df2_walls_all.empty:
        return set(), 0

    df_temp = df1_parts_all.copy()

    # Create ground surface polygons
    def coords_to_polygon(coords:list) -> Polygon:
        if not coords:
            return None
        # shapely supports 3D, pass as-is
        return Polygon(coords)

    df_temp["ground_geom"] = df_temp["ground_surface"].apply(coords_to_polygon)
    df_temp = df_temp[df_temp["ground_geom"].notnull()]

    gdf = gpd.GeoDataFrame(df_temp, geometry="ground_geom", crs=None)
    sindex = gdf.sindex

    ids_to_remove: Set[str] = set()
    same_ground_pairs = 0

    for idx, row in gdf.iterrows():
        geom = row["ground_geom"]
        if geom is None:
            continue

        # Use spatial index for candidate matches
        candidate_idx = list(sindex.intersection(geom.bounds))
        for j in candidate_idx:
            if j <= idx:
                continue
            row2 = gdf.iloc[j]
            if geom.equals(row2["ground_geom"]):
                same_ground_pairs += 1
                dup_ids = get_duplicate_wall_ids_for_two_parts(
                    df2_walls_all, row["building_id"], row2["building_id"]
                )
                ids_to_remove.update(dup_ids)

    return ids_to_remove, same_ground_pairs


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def run_pipeline(gml_dir: Path, output_dir: Path) -> None:
    """
    Full pipeline:
    - Read all .gml files in gml_dir
    - Combine into three dataframes (parts, walls, roofs)
    - Clean coordinate lists
    - Remove duplicate walls
    - Save pickles into output_dir
    """
    logging.info("Input GML directory: %s", gml_dir)
    logging.info("Output directory: %s", output_dir)

    gml_files = sorted(gml_dir.glob("*.gml"))
    if not gml_files:
        raise FileNotFoundError(f"No .gml files found in {gml_dir}")

    df1_parts_all = []
    df2_walls_all = []
    df3_roofs_all = []

    for idx, gml_file in enumerate(gml_files):
        # if idx == 2:
        #     break
        df1, df2, df3 = read_lod2_file(gml_file)
        df1_parts_all.append(df1)
        df2_walls_all.append(df2)
        df3_roofs_all.append(df3)

    df1_parts_all = pd.concat(df1_parts_all, ignore_index=True) if df1_parts_all else pd.DataFrame()
    df2_walls_all = pd.concat(df2_walls_all, ignore_index=True) if df2_walls_all else pd.DataFrame()
    df3_roofs_all = pd.concat(df3_roofs_all, ignore_index=True) if df3_roofs_all else pd.DataFrame()

    logging.info("Combined df1_parts rows: %d", len(df1_parts_all))
    logging.info("Combined df2_walls rows: %d", len(df2_walls_all))
    logging.info("Combined df3_roofs rows: %d", len(df3_roofs_all))


    # Key IDs uniqueness
    if not df1_parts_all.empty:
        logging.info("building_id uniqueness in df1: %s", df1_parts_all["building_id"].is_unique)
    if not df2_walls_all.empty:
        logging.info("wall_id uniqueness in df2: %s", df2_walls_all["wall_id"].is_unique)
    if not df3_roofs_all.empty:
        logging.info("surface_id uniqueness in df3: %s", df3_roofs_all["surface_id"].is_unique)

    # --------- optional: keep only leverkusen #FIXME  ------------------
    # logging.info("Running optional removal of addresses outside district = Leverkusen. Before %s", len(df1_parts_all))
    # df1_parts_all = df1_parts_all.loc[df1_parts_all.district == "Leverkusen"]
    # df2_walls_all = df2_walls_all[df2_walls_all["building_id"].isin(df1_parts_all["building_id"])]
    # df3_roofs_all = df3_roofs_all[df3_roofs_all["building_id"].isin(df1_parts_all["building_id"])]
    # logging.info("Running optional removal of addresses outside district = Leverkusen. After %s", len(df1_parts_all))
    # --------------------------------------------------------------------

    # Clean coordinates
    df1_parts_all = remove_double_coordinates(df1_parts_all, "ground_surface")
    df2_walls_all = remove_double_coordinates(df2_walls_all, "surface_coordinates")
    df3_roofs_all = remove_double_coordinates(df3_roofs_all, "surface_coordinates")

    # Find and remove duplicate walls
    logging.info("Searching for duplicate walls based on ground surfaces and wall geometry...")
    wall_ids_to_remove, same_ground_pairs = find_wall_ids_to_remove(df1_parts_all, df2_walls_all)

    logging.info("%d pairs of parts share the same ground surface", same_ground_pairs)
    logging.info("%d walls identified as duplicates to remove", len(wall_ids_to_remove))
    logging.info("Walls before removal: %d", len(df2_walls_all))

    if wall_ids_to_remove:
        df2_walls_all = df2_walls_all[~df2_walls_all["wall_id"].isin(wall_ids_to_remove)]
        logging.debug("Removed walls with IDs: %s", "\n".join(wall_ids_to_remove))

    logging.info("Walls after removal: %d", len(df2_walls_all))

    # Save as pickle
    df1_path = output_dir / "df1_parts.pkl"
    df2_path = output_dir / "df2_walls_00.pkl"
    df3_path = output_dir / "df3_roofs.pkl"

    df1_parts_all.to_pickle(df1_path)
    df2_walls_all.to_pickle(df2_path)
    df3_roofs_all.to_pickle(df3_path)
    
    #logging.info("Number of items in all_building.pkl: %d", len(all_building_df))
    logging.info("Saved df1_parts to %s", df1_path)
    logging.info("Saved df2_walls to %s", df2_path)
    logging.info("Saved df3_roofs to %s", df3_path)
    logging.info("Saved all_buildings.pkl to %s", output_dir / "all_buildings.pkl")
    logging.info("Pipeline finished successfully.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="LOD2 GML processing pipeline.")
    # parser.add_argument(
    #     "--gml-dir",
    #     type=str,
    #     required=True,
    #     help="Directory containing .gml files.",
    # )
    # parser.add_argument(
    #     "--output-dir",
    #     type=str,
    #     required=True,
    #     help="Directory to store output pickles.",
    # )

    # args = parser.parse_args()
    #run_pipeline(Path(args.gml_dir), Path(args.output_dir))


    run_pipeline(Path(CONF.lod2_data_dir), Path(CONF.dataframes_dir))