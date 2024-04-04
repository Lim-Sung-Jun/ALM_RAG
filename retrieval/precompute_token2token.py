import numpy as np
import faiss

embedding_dim = 512
num_queries = 2
num_kb = 3
tokens_per_audio = 1024

np.random.seed(0)
audio_query_embeddings = np.random.random((num_queries, tokens_per_audio, embedding_dim)).astype('float32')
audio_kb_embeddings = np.random.random((num_kb, tokens_per_audio, embedding_dim)).astype('float32')

audio_kb_embeddings_flat = audio_kb_embeddings.reshape(-1, embedding_dim)
batch_audio_query_embeddings = audio_query_embeddings.reshape(-1, embedding_dim)

index = faiss.IndexFlatL2(embedding_dim)
index.add(audio_kb_embeddings_flat)

top_k = tokens_per_audio * num_kb # 모든 결과가 있어야 token2token이 된다.
D, I = index.search(batch_audio_query_embeddings, top_k)

# 마지막에 모든 query와 document 점수 비교
final_score_matrix = np.zeros((num_queries, num_kb))


I = I.reshape(num_queries, tokens_per_audio, top_k)
D = D.reshape(num_queries, tokens_per_audio, top_k)

# 각 오디오 토큰이 어떤 오디오에 속하는지 알기 위해서 인덱스를 나눠줌
doc_indices = I // tokens_per_audio # 

# 점수 계산
for doc_id in range(num_kb):
    # 모든 토큰에 대해서 현재 오디오와 관련된 쿼리 토큰만 true (현재 오디오만 신경쓴다는 소리)
    mask = (doc_indices == doc_id)

    # distance에 현재 오디오와 관련없는 것을 마스킹
    masked_distances = np.where(mask, -D, -np.inf)

    # 모든 쿼리와 현재 오디오의 토큰 사이의 점수 중에서 최대값만 남겨둔다. (모든 토큰에 대해서 s1, s2, ..., s1024 구함)
    max_scores = np.max(masked_distances, axis=2)
    # 각 쿼리마다 최대값을 더한다. (쿼리의 1024개 토큰에서 최대값을 쭉 더한다.) (final_score = s1 + s2 + ... + s1024)
    final_score_matrix[:, doc_id] = np.sum(max_scores, axis=1)

# Displaying the final score matrix
print("Final Score Matrix:")
print(final_score_matrix)