import copy
import time
import json
import requests
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify
from utils.parse_frontend import parse_data
from utils.faiss_processing import MyFaiss
from utils.context_encoding import VisualEncoding
from utils.semantic_embed.tag_retrieval import tag_retrieval
from utils.combine_utils import merge_searching_results_by_addition
from utils.search_utils import group_result_by_video, search_by_filter

json_path = 'E:/AIO-2022 - Copy/Competition/Competition_AIChallenge2023/AIChallenge2023/dict/id2img_fps.json'
audio_json_path = 'E:/AIO-2022 - Copy/Competition/Competition_AIChallenge2023/AIChallenge2023/dict/audio_id2img_id.json'
scene_path = 'E:/AIO-2022 - Copy/Competition/Competition_AIChallenge2023/AIChallenge2023/dict/scene_id2info.json'
bin_clip_file ='E:/AIO-2022 - Copy/Competition/Competition_AIChallenge2023/AIChallenge2023/dict/faiss_clip_cosine.bin'
bin_clipv2_file ='E:/AIO-2022 - Copy/Competition/Competition_AIChallenge2023/AIChallenge2023/dict/faiss_clipv2_cosine.bin'
video_division_path = 'E:/AIO-2022 - Copy/Competition/Competition_AIChallenge2023/AIChallenge2023/dict/video_division_tag.json'
img2audio_json_path = 'E:/AIO-2022 - Copy/Competition/Competition_AIChallenge2023/AIChallenge2023/dict/img_id2audio_id.json'

# Tạo ra lưới để thêm các icon vào (Chức năng của Component A)
VisualEncoder = VisualEncoding()
# Sử dụng để gọi các phương thức khác nhau của lớp MyFaiss để thực hiện các truy vấn tìm kiếm hình ảnh, văn bản, OCR, ASR, và các chức năng khác.
CosineFaiss = MyFaiss(bin_clip_file, bin_clipv2_file, json_path, audio_json_path, img2audio_json_path)
# Khởi tạo và cấu hình đối tượng tag_retrieval với các tham số cần thiết, 
# chuẩn bị cho các hoạt động gợi ý thẻ (tag recommendation) dựa trên các nhúng ngữ nghĩa được trích xuất từ văn bản.
TagRecommendation = tag_retrieval()
# DictImagePath: Lưu trữ từ điển ánh xạ giữa chỉ số và thông tin đường dẫn ảnh.
# TotalIndexList: Danh sách các chỉ số của tất cả các ảnh, được chuyển thành mảng NumPy với kiểu dữ liệu int64.
DictImagePath = CosineFaiss.id2img_fps
TotalIndexList = np.array(list(range(len(DictImagePath)))).astype('int64')

with open(scene_path, 'r') as f:
  Sceneid2info = json.load(f)

with open('E:/AIO-2022 - Copy/Competition/Competition_AIChallenge2023/AIChallenge2023/dict/map_keyframes.json', 'r') as f:
  KeyframesMapper = json.load(f)

with open(video_division_path, 'r') as f:
  VideoDivision = json.load(f)

with open('E:/AIO-2022 - Copy/Competition/Competition_AIChallenge2023/AIChallenge2023/dict/video_id2img_id.json', 'r') as f:
  Videoid2imgid = json.load(f)

# Lấy danh sách các ảnh liên quan đến các video thuộc id được chỉ định.
def get_search_space(id):
  # id starting from 1 to 4
  search_space = []
  video_space = VideoDivision[f'list_{id}']
  for video_id in video_space:
    search_space.extend(Videoid2imgid[video_id])
  return search_space

# Tạo từ điển chứa các không gian tìm kiếm dựa trên các id từ 1 đến 4.
SearchSpace = dict()
for i in range(1, 5):
  SearchSpace[i] = np.array(get_search_space(i)).astype('int64')
SearchSpace[0] = TotalIndexList

# Lấy danh sách các khung hình gần kề với khung hình có chỉ số idx
def get_near_frame(idx):
  image_info = DictImagePath[idx]
  scene_idx = image_info['scene_idx'].split('/')
  near_keyframes_idx = copy.deepcopy(Sceneid2info[scene_idx[0]][scene_idx[1]][scene_idx[2]][scene_idx[3]]['lst_keyframe_idxs'])
  return near_keyframes_idx

# Lấy danh sách các khung hình cần bỏ qua dựa trên danh sách các chỉ mục cần bỏ qua ban đầu.
def get_related_ignore(ignore_index):
  total_ignore_index = []
  for idx in ignore_index:
    total_ignore_index.extend(get_near_frame(idx))
  return total_ignore_index

# Run Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)

# # Run Flask app
# app = Flask(__name__, template_folder='templates')
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
'''
Chức năng cụ thể của các hàm được sử dụng:
- CosineFaiss.image_search: Thực hiện tìm kiếm hình ảnh dựa trên embedding cosine similarity.
- CosineFaiss.text_search: Thực hiện tìm kiếm văn bản dựa trên embedding cosine similarity.
- group_result_by_video: Nhóm kết quả tìm kiếm theo video.
- CosineFaiss.context_search: Tìm kiếm dựa trên bối cảnh bao gồm đối tượng, OCR, ASR.
- TagRecommendation: Đưa ra gợi ý thẻ từ truy vấn văn bản.
- CosineFaiss.reranking: Xếp hạng lại kết quả tìm kiếm dựa trên phản hồi của người dùng.
- CosineFaiss.translater: Dịch truy vấn văn bản.
'''

# Trả về dữ liệu trang chứa các hình ảnh và ID tương ứng từ DictImagePath. 
# Chỉ lấy các ID nhỏ hơn hoặc bằng 500.
@app.route('/data')
def index():
    pagefile = []
    for id, value in DictImagePath.items():
        if int(id) > 500:
          break
        pagefile.append({'imgpath': value['image_path'], 'id': id})
    data = {'pagefile': pagefile}
    return jsonify(data)

# Thực hiện tìm kiếm hình ảnh dựa trên ID của hình ảnh và số lượng kết quả (k). 
# Sử dụng phương pháp tìm kiếm Cosine Faiss để trả về danh sách các điểm số, ID, 
# và đường dẫn hình ảnh tương ứng.
@app.route('/imgsearch')
def image_search():
    print("image search")
    k = int(request.args.get('k'))
    id_query = int(request.args.get('imgid'))
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.image_search(id_query, k=k)

    data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)

    return jsonify(data)

# Thực hiện tìm kiếm văn bản dựa trên truy vấn văn bản (textquery) và các tham số khác từ yêu cầu POST.
# Sử dụng các mô hình CLIP và FAISS để tìm kiếm và trả về kết quả dựa trên điểm số và ID.
@app.route('/textsearch', methods=['POST'], strict_slashes=False)
def text_search():
    print("text search")
    data = request.json
    
    print(">>text seach data")
    print(data)
    search_space_index = int(data['search_space'])
    print(">> search_space_index", search_space_index)

    k = int(data['k'])
    print(">> k", k)

    clip = data['clip']
    print(">> clip", clip)

    clipv2 = data['clipv2']
    print(">> clipv2", clipv2)

    text_query = data['textquery']
    print(">> text_query", text_query)

    range_filter = int(data['range_filter'])
    print(">> range_filter", range_filter)

    index = None
    if data['filter']:
      index = np.array(data['id']).astype('int64')
      k = min(k, len(index))
      print("using index")


    keep_index = None
    ignore_index = None
    if data['ignore']:
      ignore_index = get_related_ignore(np.array(data['ignore_idxs']).astype('int64'))
      keep_index = np.delete(TotalIndexList, ignore_index)
      print("using ignore")
    print(">>>>>> kepindex",keep_index)
    print(">>>>>> ignoreindex",ignore_index)
    print(">>>>>> index",index)


    if keep_index is not None:
      if index is not None:
        index = np.intersect1d(index, keep_index)
      else:
        index = keep_index

    if index is None:
      index = SearchSpace[search_space_index]
    else:
      index = np.intersect1d(index, SearchSpace[search_space_index])
    k = min(k, len(index))

    if clip and clipv2:
      model_type = 'both'
    elif clip:
       model_type = 'clip'
    else:
       model_type = 'clipv2'

    if data['filtervideo'] != 0:
      print('filter video')
      mode = data['filtervideo']
      prev_result = data['videos']
      data = search_by_filter(prev_result, text_query, k, mode, model_type, range_filter, ignore_index, keep_index, Sceneid2info, DictImagePath, CosineFaiss, KeyframesMapper)
    else:
      if model_type == 'both':
        scores_clip, list_clip_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clip')
        scores_clipv2, list_clipv2_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clipv2')
        lst_scores, list_ids = merge_searching_results_by_addition([scores_clip, scores_clipv2],
                                                                  [list_clip_ids, list_clipv2_ids])
        infos_query = list(map(CosineFaiss.id2img_fps.get, list(list_ids)))
        list_image_paths = [info['image_path'] for info in infos_query]
      else:
        lst_scores, list_ids, _, list_image_paths = CosineFaiss.text_search(text_query, index=index, k=k, model_type=model_type)
      data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)
    print(">>>>>data")
    # print(1)
    print(data)
    return jsonify(data)

# @app.route('/textsearch', methods=['POST'], strict_slashes=False)
# def text_search():
#     print("text search")
#     data = request.json
    
#     # Extract and print input data
#     search_space_index = int(data['search_space'])
#     k = int(data['k'])
#     clip = data['clip']
#     clipv2 = data['clipv2']
#     text_query = data['textquery']
#     range_filter = int(data['range_filter'])
    
#     index = None
#     if data.get('filter'):
#         index = np.array(data['id']).astype('int64')
#         k = min(k, len(index))
#         print("using index")

#     keep_index, ignore_index = None, None
#     if data.get('ignore'):
#         ignore_index = get_related_ignore(np.array(data['ignore_idxs']).astype('int64'))
#         keep_index = np.delete(TotalIndexList, ignore_index)
#         print("using ignore")

#     print(">>>>>> keep_index", keep_index)
#     print(">>>>>> ignore_index", ignore_index)
#     print(">>>>>> index", index)

#     # Combine indices if needed
#     if keep_index is not None:
#         if index is not None:
#             index = np.intersect1d(index, keep_index)
#         else:
#             index = keep_index

#     if index is None:
#         index = SearchSpace[search_space_index]
#     else:
#         index = np.intersect1d(index, SearchSpace[search_space_index])
    
#     k = min(k, len(index))

#     # Determine model type
#     if clip and clipv2:
#         model_type = 'both'
#     elif clip:
#         model_type = 'clip'
#     else:
#         model_type = 'clipv2'

#     if data['filtervideo'] != 0:
#         print('filter video')
#         mode = data['filtervideo']
#         prev_result = data['videos']
#         data = search_by_filter(
#             prev_result, text_query, k, mode, model_type, range_filter, ignore_index, keep_index, 
#             Sceneid2info, DictImagePath, CosineFaiss, KeyframesMapper
#         )
#     else:
#         if model_type == 'both':
#             scores_clip, list_clip_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clip')
#             scores_clipv2, list_clipv2_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clipv2')
#             lst_scores, list_ids = merge_searching_results_by_addition([scores_clip, scores_clipv2], [list_clip_ids, list_clipv2_ids])
#             infos_query = list(map(CosineFaiss.id2img_fps.get, list(list_ids)))
#             list_image_paths = [info['image_path'] for info in infos_query]
#         else:
#             lst_scores, list_ids, _, list_image_paths = CosineFaiss.text_search(text_query, index=index, k=k, model_type=model_type)
        
#         data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)
    
#     print(">>>>>data")
#     print(data)
#     return jsonify(data)


# Thực hiện tìm kiếm dựa trên nhiều yếu tố khác nhau như đối tượng, 
# OCR (văn bản viết tay), ASR (nhận dạng giọng nói tự động). 
# Kết quả tìm kiếm được trả về dưới dạng danh sách các điểm số và ID hình ảnh tương ứng.
@app.route('/panel', methods=['POST'], strict_slashes=False)
def panel():
    print("panel search")
    search_items = request.json
    k = int(search_items['k'])
    search_space_index = int(search_items['search_space'])

    index = None
    if search_items['useid']:
      index = np.array(search_items['id']).astype('int64')
      k = min(k, len(index))

    keep_index = None
    if search_items['ignore']:
      ignore_index = get_related_ignore(np.array(search_items['ignore_idxs']).astype('int64'))
      keep_index = np.delete(TotalIndexList, ignore_index)
      print("using ignore")

    if keep_index is not None:
      if index is not None:
        index = np.intersect1d(index, keep_index)
      else:
        index = keep_index

    if index is None:
      index = SearchSpace[search_space_index]
    else:
      index = np.intersect1d(index, SearchSpace[search_space_index])
    k = min(k, len(index))

    # Parse json input
    object_input = parse_data(search_items, VisualEncoder)
    if search_items['ocr'] == "":
      ocr_input = None
    else:
      ocr_input = search_items['ocr']

    if search_items['asr'] == "":
      asr_input = None
    else:
      asr_input = search_items['asr']

    semantic = False
    keyword = True
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.context_search(object_input=object_input, ocr_input=ocr_input, asr_input=asr_input,
                                                                           k=k, semantic=semantic, keyword=keyword, index=index, useid=search_items['useid'])

    data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)
    return jsonify(data)

# Lấy gợi ý thẻ từ truy vấn văn bản. Trả về danh sách các gợi ý thẻ (tag) dựa trên truy vấn văn bản.
# Người dùng có thể dùng cái này để tăng tính chính xác cho truy vấn (Cái này có vẻ không cần thiết)
@app.route('/getrec', methods=['POST'], strict_slashes=False)
def getrec():
    print("get tag recommendation")
    k = 50
    text_query = request.json
    print("req>>>>>>>>>>:")
    print(text_query)

    tag_outputs = TagRecommendation(text_query, k)
    print("2")
    return jsonify(tag_outputs)

# Lấy thông tin hình ảnh liên quan dựa trên ID của hình ảnh. 
# Trả về URL video, khoảng thời gian của video và các khung hình chính liên quan.
@app.route('/relatedimg')
def related_img():
    print("related image")
    id_query = int(request.args.get('imgid'))
    image_info = DictImagePath[id_query]
    image_path = image_info['image_path']
    scene_idx = image_info['scene_idx'].split('/')

    video_info = copy.deepcopy(Sceneid2info[scene_idx[0]][scene_idx[1]])
    video_url = video_info['video_metadata']['watch_url']
    video_range = video_info[scene_idx[2]][scene_idx[3]]['shot_time']

    near_keyframes = video_info[scene_idx[2]][scene_idx[3]]['lst_keyframe_paths']
    near_keyframes.remove(image_path)

    data = {'video_url': video_url, 'video_range': video_range, 'near_keyframes': near_keyframes}
    return jsonify(data)

# Lấy các đoạn video liên quan dựa trên ID của hình ảnh. 
# Trả về thông tin đoạn video, bao gồm các khung hình chính và chỉ số đoạn video đã chọn.
@app.route('/getvideoshot')
def get_video_shot():
    print("get video shot")

    if request.args.get('imgid') == 'undefined':
      return jsonify(dict())

    id_query = int(request.args.get('imgid'))
    image_info = DictImagePath[id_query]
    scene_idx = image_info['scene_idx'].split('/')
    shots = copy.deepcopy(Sceneid2info[scene_idx[0]][scene_idx[1]][scene_idx[2]])

    selected_shot = int(scene_idx[3])
    total_n_shots = len(shots)
    new_shots = dict()
    for select_id in range(max(0, selected_shot-5), min(selected_shot+6, total_n_shots)):
      new_shots[str(select_id)] = shots[str(select_id)]
    shots = new_shots

    for shot_key in shots.keys():
      lst_keyframe_idxs = []
      for img_path in shots[shot_key]['lst_keyframe_paths']:
        data_part, video_id, frame_id = img_path.replace('/data/KeyFrames/', '').replace('.webp', '').split('/')
        key = f'{data_part}_{video_id}'.replace('_extra', '')
        if 'extra' not in data_part:
          frame_id = KeyframesMapper[key][str(int(frame_id))]
        frame_id = int(frame_id)
        lst_keyframe_idxs.append(frame_id)
      shots[shot_key]['lst_idxs'] = shots[shot_key]['lst_keyframe_idxs']
      shots[shot_key]['lst_keyframe_idxs'] = lst_keyframe_idxs

    data = {
        'collection': scene_idx[0],
        'video_id': scene_idx[1],
        'shots': shots,
        'selected_shot': scene_idx[3]
    }
    return jsonify(data)

# Xử lý phản hồi từ người dùng dựa trên các kết quả tìm kiếm trước đó. 
# Thực hiện xếp hạng lại các kết quả dựa trên phản hồi tích cực và tiêu cực.
@app.route('/feedback', methods=['POST'], strict_slashes=False)
def feed_back():
    data = request.json
    k = int(data['k'])
    prev_result = data['videos']
    lst_pos_vote_idxs = data['lst_pos_idxs']
    lst_neg_vote_idxs = data['lst_neg_idxs']
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.reranking(prev_result, lst_pos_vote_idxs, lst_neg_vote_idxs, k)
    data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)
    return jsonify(data)

# Dịch truy vấn văn bản sang ngôn ngữ khác. Trả về truy vấn văn bản đã được dịch.
@app.route('/translate', methods=['POST'], strict_slashes=False)
def translate():
  data = request.json
  text_query = data['textquery']
  text_query_translated = CosineFaiss.translater(text_query)
  return jsonify(text_query_translated)

# Running app
if __name__ == '__main__':
    app.run(debug=True, port=8080)