def retained_positions(length,image_start,dropped,visual_tokens=576):
    if image_start<0 or image_start+visual_tokens>length or len(dropped)!=len(set(dropped)) or any(type(i) is not int or not 0<=i<visual_tokens for i in dropped):
        raise ValueError('Invalid visual token discard IDs')
    removed={image_start+i for i in dropped}
    return [i for i in range(length) if i not in removed]
