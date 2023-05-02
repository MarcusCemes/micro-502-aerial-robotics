from asyncio import create_task, gather, run, to_thread
from enum import Enum
from queue import Queue
from threading import Thread
from typing import Any, Set, Tuple

from aiohttp import WSMsgType, web
from common import Context
from config import SERVER_PORT
from log import Logger

QUEUE_SIZE = 32


class MessageType(Enum):
    Close = 0
    Broadcast = 1


Message = Tuple[MessageType, Any]


class WebApplication(web.Application, Logger):
    def __init__(self, queue: Queue[Message]):
        super().__init__()

        self.queue = queue
        self.clients: Set[web.WebSocketResponse] = set()

        self.add_routes([web.get("/", self.hello)])
        self.add_routes([web.get("/ws", self.websocket_handler)])

    async def run(self, port=SERVER_PORT):
        print("Starting server...")

        runner = web.AppRunner(self)
        await runner.setup()

        site = web.TCPSite(runner, "localhost", port)

        await site.start()
        self.info(f"Server started at http://localhost:{port}")

        try:
            while True:
                msg = await to_thread(self.queue.get)
                await self.handle_message(msg)

        except InterruptedError:
            pass

        self.info("Stopping server...")
        for client in self.clients:
            await client.close()

        await runner.cleanup()
        self.info("Server stopped")

    async def handle_message(self, msg: Message):
        match msg:
            case (MessageType.Close, None):
                raise InterruptedError

            case (MessageType.Broadcast, payload):
                for client in self.clients:
                    try:
                        await client.send_json(payload)
                    except Exception as e:
                        self.logger.warning(f"Error sending message: {e}")

    async def hello(self, *_):
        return web.Response(text="Drone Controller Server")

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.clients.add(ws)

        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                print(f"WebSocket error: {ws.exception()}")

        self.clients.remove(ws)

        return ws


class Server(Thread, Logger):
    def __init__(self, ctx: Context):
        super().__init__()

        self.ctx = ctx
        self.queue: Queue[Message] = Queue(QUEUE_SIZE)

    def start(self):
        super().start()
        self.ctx.outlet.subscribe(self.on_message)

    def stop(self):
        self.queue.put((MessageType.Close, None))

    def run(self):
        app = WebApplication(self.queue)
        co = app.run()
        run(co)

    def on_message(self, msg):
        self.queue.put((MessageType.Broadcast, msg))
