import grpc 
from concurrent import futures
from grpc_pkg.comment_pb import comment_censor_pb2
from grpc_pkg.comment_pb import comment_censor_pb2_grpc
from models.text_moderator import TextModerator
from models.image_moderator import ImageModerator
from utils.config_reader import ConfigReader

class CommentCensorServicer(comment_censor_pb2_grpc.CommentCensorServiceServicer):
    def __init__(self):
        self.text_moderator = TextModerator(cache_dir="cache")
        self.image_moderator = ImageModerator()

    def CheckComment(self, request, context):
        # comment_id = request.comment_id
        content = request.content

        result = self.text_moderator.moderate(content)

        return comment_censor_pb2.CommentCensorResponse(censor_content=result["censored_text"])
    def CheckImage(self, request, context):
        base_url = request.base_url
        media_file = request.media_file
        result = self.image_moderator.moderate(base_url, media_file)
        return comment_censor_pb2.ImageCensorResponse(label=result["label"], score=result.get("score", 0.0))

def serve(config_env=None):
    config_reader = ConfigReader(config_env)
    comment_censor_grpc_conn = config_reader.get_comment_censor_grpc_conn_config()
    host = comment_censor_grpc_conn.host
    port = comment_censor_grpc_conn.port

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    comment_censor_pb2_grpc.add_CommentCensorServiceServicer_to_server(CommentCensorServicer(), server)

    bind_address = f"{host}:{port}"
    server.add_insecure_port(bind_address)
    print(f"gRPC server started on {bind_address}...")
    server.start()
    server.wait_for_termination()
