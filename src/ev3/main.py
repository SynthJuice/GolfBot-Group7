#!/usr/bin/env python3
import socket

from motor_control import MotorController
from command_handler import CommandHandler

HOST = "0.0.0.0"
PORT = 5000
RECV_CHUNK_SIZE = 8192
SUCCESS_RESPONSE = b"EV3 got command\n"


def loop(conn, command_handler, motor_controller):
    buffer = bytearray()

    while True:
        chunk = conn.recv(RECV_CHUNK_SIZE)

        if not chunk:
            print("Client disconnected")
            motor_controller.stop()
            break

        buffer.extend(chunk)

        while buffer:
            expected_length = command_handler.get_expected_length(buffer)

            if expected_length is None:
                break

            if len(buffer) < expected_length:
                break

            command_bytes = bytes(buffer[:expected_length])
            del buffer[:expected_length]

            if not command_handler.handle_command(command_bytes):
                motor_controller.stop()
                return

            print("Received raw bytes length:", len(command_bytes))
            print("Received raw bytes preview:", list(command_bytes[:40]))

            conn.sendall(SUCCESS_RESPONSE)


def main():
    motor_controller = MotorController()
    command_handler = CommandHandler(motor_controller)

    supported_codes = ["0x{:X}".format(code) for code in sorted(command_handler.commands.keys())]
    print("Supported command codes:", supported_codes)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        print("Listening on {}:{}...".format(HOST, PORT))

        conn, addr = server.accept()
        with conn:
            print("Connected by {}".format(addr))
            loop(conn, command_handler, motor_controller)


if __name__ == "__main__":
    main()
