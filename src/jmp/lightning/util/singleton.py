"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from logging import getLogger
from typing import Any

from typing_extensions import Self, TypeVar, override

log = getLogger(__name__)


class Singleton:
    singleton_key = "_singleton_instance"

    @classmethod
    def get(cls) -> Self | None:
        return getattr(cls, cls.singleton_key, None)

    @classmethod
    def set(cls, instance: Self) -> None:
        if cls.get() is not None:
            log.warning(f"{cls.__qualname__} instance is already set")

        setattr(cls, cls.singleton_key, instance)

    @classmethod
    def reset(cls) -> None:
        if cls.get() is not None:
            delattr(cls, cls.singleton_key)

    @classmethod
    def register(cls, instance: Self) -> None:
        cls.set(instance)

    @classmethod
    def instance(cls) -> Self:
        instance = cls.get()
        if instance is None:
            raise RuntimeError(f"{cls.__qualname__} instance is not set")

        return instance

    @override
    def __init_subclass__(cls, *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)

        cls.reset()


T = TypeVar("T", infer_variance=True)


class Registry:
    _registry: dict[type, Any] = {}

    @staticmethod
    def register(cls_: type[T], instance: T):
        if not isinstance(instance, cls_):
            raise ValueError(f"{instance} is not an instance of {cls_.__qualname__}")

        if cls_ in Registry._registry:
            raise ValueError(f"{cls_.__qualname__} is already registered")

        Registry._registry[cls_] = instance

    @staticmethod
    def try_get(cls_: type[T]) -> T | None:
        return Registry._registry.get(cls_)

    @staticmethod
    def get(cls_: type[T]) -> T:
        instance = Registry.try_get(cls_)
        if instance is None:
            raise ValueError(f"{cls_.__qualname__} is not registered")

        return instance

    @staticmethod
    def instance(cls_: type[T]) -> T:
        return Registry.get(cls_)

    @staticmethod
    def reset(cls_: type[T]):
        if cls_ in Registry._registry:
            del Registry._registry[cls_]

    @staticmethod
    def reset_all():
        Registry._registry.clear()
