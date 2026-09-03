
import numpy as np
from PIL import Image
import arabic_reshaper
from bidi.algorithm import get_display
def np_to_pil(img: np.ndarray):
    #Numpy to PIL Image
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = Image.fromarray(img, mode="RGBA").convert("RGB")
    else:
        img = Image.fromarray(img).convert("RGB")
    return img



def reshape_position_ids_for_rtl(position_ids):
    cols = [position_id[1] for position_id in position_ids]
    rows = [position_id[0] for position_id in position_ids]
    new_position_ids = []

    min_col_per_row = {row: min(cols[i] for i, r in enumerate(rows) if r == row) for row in set(rows)}
    max_col_per_row = {row: max(cols[i] for i, r in enumerate(rows) if r == row) for row in set(rows)}
    
    max_col_per_row = {row: max(cols[i] for i, r in enumerate(rows) if r == row) for row in set(rows)}
    
    for i, col in enumerate(cols):
        row = position_ids[i][0]
        new_col = max_col_per_row[row] + min_col_per_row[row] - col  # flip
        new_position_ids.append((row, new_col))
    
    return new_position_ids

def present_rtl(text):
    return get_display(arabic_reshaper.reshape(text))

if __name__ == "__main__":
    #print(generate_sent_idx("""Wer an die Ostsee reist, loggt sich innerlich aus. Ganz gleich ob am Timmendorfer Strand, in Binz auf Rügen, in den Stockholmer Schärengärten, im polnischen Sopot, im lettischen Jūrmala oder im estnischen Tallinn: Synchronisiert man sich gedanklich mit dem Jauchzen der Silbermöwen und blickt auf den baltischen Horizont, überfällt einen schnell das finale Feriengefühl: Mehr braucht's nicht.""".split(" "), lang = "de"))
    position_ids = [(1,0),(1,1),(1,2)]
    print(reshape_position_ids_for_rtl(position_ids))