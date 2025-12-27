#python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. your_file.proto
import os


def gen_service():
    proto_path = "task_service.proto"
    out_path = "."
    # execute cmd in terminal
    cmd = f"python -m grpc_tools.protoc -I. --python_out={out_path} --grpc_python_out={out_path} {proto_path}"
    os.system(cmd)

if __name__ == '__main__':
    gen_service()
    print("Service generated")