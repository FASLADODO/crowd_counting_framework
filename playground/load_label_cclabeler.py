import json

path = "/data/my_crowd_image/street_20201127/data_batch2/jsons"


if __name__ == "__main__":
    a_json = "/data/my_crowd_image/street_20201127/data_batch2/jsons/IMG_20201127_161346_841.json"
    json_data = json.load(open(a_json))
    print("human count", json_data['human_num'])
    print("list of points")
    pts = json_data['points']
    for pt in pts:
        print("a point ", pt, type(pt))