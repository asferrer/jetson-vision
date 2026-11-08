# Per-frame metadata that does not fit in vision_msgs/Detection2DArray.
# The header is copied from the input image, like the detections.
std_msgs/Header header
uint32 schema_version
string model_id
string session_id
uint64 frame_seq
float32 score_threshold
uint8 degradation_level
uint32 dropped_before
uint64 capture_ns
uint64 preprocess_ns
uint64 inference_ns
uint64 postprocess_ns
uint64 tracking_ns
uint64 total_ns
uint64 queue_ns
