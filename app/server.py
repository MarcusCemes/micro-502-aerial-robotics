from asyncio import Event, create_task
from typing import Any, Set

from aiohttp import WSMsgType, web
from loguru import logger

from .common import Context
from .config import MAP_PX_PER_M, MAP_SIZE, SERVER_PORT
from .utils.observable import Observable


class WebApplication(web.Application):
    def __init__(self, outlet: Observable, stop: Event):
        super().__init__()

        self._clients: Set[web.WebSocketResponse] = set()
        self._outlet = outlet
        self._stop = stop

        self.add_routes([web.get("/", self.hello)])
        self.add_routes([web.get("/ws", self.websocket_handler)])

    async def run(self, port=SERVER_PORT):
        logger.info("Starting server...")

        runner = web.AppRunner(self)
        await runner.setup()

        site = web.TCPSite(runner, "localhost", port)

        await site.start()
        logger.info(f"Server started at http://localhost:{port}")

        await self._stop.wait()

        logger.info("Stopping server...")
        for client in self._clients:
            await client.close()

        await runner.cleanup()
        logger.info("Server stopped")

    async def hello(self, *_):
        return web.Response(text="Drone Controller Server")

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        await self.send_config(ws)

        def callback(msg: Any):
            create_task(ws.send_json(msg))

        self._outlet.subscribe(callback)

        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")

        self._outlet.unregister(callback)

        return ws

    async def send_config(self, ws: web.WebSocketResponse):
        await ws.send_json(
            {
                "type": "config",
                "data": {
                    "MAP_PX_PER_M": MAP_PX_PER_M,
                    "MAP_SIZE": list(MAP_SIZE),
                },
            }
        )


class Server:
    def __init__(self, ctx: Context):
        super().__init__()

        self.ctx = ctx

    async def run(self, stop: Event):
        app = WebApplication(self.ctx.outlet, stop)

        await app.run()
