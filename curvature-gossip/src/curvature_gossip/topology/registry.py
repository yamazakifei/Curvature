"""维护拓扑生成器注册表，使调用方无需拓扑类型条件分支。"""

from typing import Callable, Dict, Type

from .base import TopologyGenerator


_REGISTRY = {}  # type: Dict[str, Type[TopologyGenerator]]


def register_topology(name: str) -> Callable[[Type[TopologyGenerator]], Type[TopologyGenerator]]:
    def decorator(cls: Type[TopologyGenerator]) -> Type[TopologyGenerator]:
        if not name or name in _REGISTRY:
            raise ValueError("拓扑名称为空或重复注册: {}".format(name))
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_topology_generator(name: str) -> TopologyGenerator:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            "未知拓扑类型 '{}'; 可用类型: {}".format(name, ", ".join(sorted(_REGISTRY)))
        )


def registered_topologies():
    return tuple(sorted(_REGISTRY))

