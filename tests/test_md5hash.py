import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import hashlib

def calculate_md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()



def test_file_integrity():
    filename = "./data/slideseq/celloracle_links_day3_1.pkl"
    md5_hash = calculate_md5(filename)
    assert md5_hash == "a1f83312e45e8458f05801f18e37a01a"

    filename = "./data/slideseq/celloracle_links_day3_2.pkl"
    md5_hash = calculate_md5(filename)
    assert md5_hash == "75fb49e777f886e0f1b150193f4d7930"

    filename = "./data/slideseq/celltype_assign.json"
    md5_hash = calculate_md5(filename)
    assert md5_hash == "2795a6f2b487dee5cd5632162551f1c4"

    



    
