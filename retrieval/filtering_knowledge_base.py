import json

# Function to load a JSON file
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Load the retrieved results
retrieved_data = load_json('/home/sungjun/projects/ALM_RAG/audio&text2audio&text_QclothoVal_KBlargeAudio&Text/audio&text_2caption2audio&text_2caption.json')

# Load the training data from each dataset
audioset_data = load_json('/drl_nas1/ckddls1321/dev/DL/AudioRAG/data/json_files/AudioSet/train.json')
clotho_data = load_json('/drl_nas1/ckddls1321/dev/DL/AudioRAG/data/json_files/Clotho/train.json')  # Update with actual path
wavcaps_data1 = load_json('/drl_nas1/ckddls1321/dev/DL/AudioRAG/data/json_files/BBC_Sound_Effects/bbc_final.json')  # Update with actual path
wavcaps_data2 = load_json('/drl_nas1/ckddls1321/dev/DL/AudioRAG/data/json_files/FreeSound/fsd_final.json')
wavcaps_data3 = load_json('/drl_nas1/ckddls1321/dev/DL/AudioRAG/data/json_files/SoundBible/sb_final.json')
wavcaps_data4 = load_json('/drl_nas1/ckddls1321/dev/DL/AudioRAG/data/json_files/AudioSet_SL/as_final.json')

# Consolidate all training data into a single dictionary for easier lookup
all_train_data = {}
for entry in audioset_data['data']:
    all_train_data[entry['audio']] = entry
for entry in clotho_data['data']:
    all_train_data[entry['audio']] = entry
for entry in wavcaps_data1['data']:
    all_train_data[entry['audio']] = entry
for entry in wavcaps_data2['data']:
    all_train_data[entry['audio']] = entry
for entry in wavcaps_data3['data']:
    all_train_data[entry['audio']] = entry
for entry in wavcaps_data4['data']:
    all_train_data[entry['audio']] = entry

# Filter and map the data
filtered_data = {'num_captions_per_audio': 1, 'data': []}
for query_wavpath, top_results in retrieved_data.items():
    for wavpath, _ in top_results:
        if wavpath in all_train_data:
            filtered_data['data'].append(all_train_data[wavpath])

# Write the filtered data to a new JSON file
with open('filtered_kb.json', 'w') as file:
    json.dump(filtered_data, file, indent=4)

print("Filtered data saved to 'filtered/train.json'")