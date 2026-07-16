"""为未来分布式 ORC 协议保留明确不可运行的版本 1 占位类。"""

from .base import DistributedCurvatureAgent


class DistributedORCAgent(DistributedCurvatureAgent):
    def _unsupported(self):
        raise NotImplementedError("Distributed ORC control messages are not implemented in version 1")

    initialize = lambda self, node_id, local_view: self._unsupported()
    create_messages = lambda self: self._unsupported()
    receive = lambda self, message: self._unsupported()
    is_ready = lambda self: self._unsupported()
    incident_curvatures = lambda self: self._unsupported()

