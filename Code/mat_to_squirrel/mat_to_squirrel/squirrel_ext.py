from typing import Optional
import fsspec
import msgpack
import msgpack_numpy
from squirrel.driver.store_driver import StoreDriver
from squirrel.serialization.msgpack import MessagepackSerializer

from enum import Enum

class Split(str, Enum):
    train = "train"
    validation = "validation"
    test = "test"


# When writing the dataset, it would be easiest
# to ensure that the dataset has the desired output format
class ConfigurableMessagePackDriver(StoreDriver):
    name = "configurable_messagepack"

    def __init__(self, url: str, **kwargs):
        """Initializes MessagepackDriver with default store and custom serializer."""
        if "store" in kwargs:
            raise ValueError(
                "Store of MessagepackDriver is fixed, `store` cannot be provided."
            )
        # This happens in top-super and nothing else:
        # self._catalog = catalog if catalog is not None else Catalog()
        # and
        # self._store = store is set
        super().__init__(url, ConfigurableMessagepackSerializer, **kwargs)

    # get_iter of super calls nothing else than self.get()
    # here it would use a sharded access store
    # Now, I am thinking that the SquirrelStore should be updated to include the split variable
    # Then the get/set methods of the SquirrelStore could include the split name
    # and auto prepend it... But where? And how?
    # I would have to write a custom get/set method that calls the super variant
    # Or... I could simply name the keys depending on the split
    # I think this would be the easiest approach
    def get_iter(
        self,
        split: Optional[Split] = None,
        **kwargs,
    ):
        """
        Returns an iterable of items in the form of a :py:class:`squirrel.iterstream.Composable`, which allows
        various stream manipulation functionalities.

        Items are fetched using the :py:meth:`get` method. The returned :py:class:`Composable` iterates over the items
        in the order of the keys returned by the :py:meth:`keys` method.

        Args:
            flatten (bool): Whether to flatten the returned iterable. Defaults to True.
            **kwargs: Other keyword arguments passed to `super().get_iter()`. For details, see
                :py:meth:`squirrel.driver.MapDriver.get_iter`.

        Returns:
            (squirrel.iterstream.Composable) Iterable over the items in the store.

        See map-driver.get_iter!
        """
        keys = super().keys()
        if split is not None:
            keys = [k for k in keys if k.startswith(split)]
        return super().get_iter(keys_iterable=keys, **kwargs)


class ConfigurableMessagepackSerializer(MessagepackSerializer):
    """
    Identical to `MessagepackSerializer` with only difference
    that I allow custom compression method!

    To minimize the required custom code, the data will
    still be suffixed with `.gz`, irrespective of the used compression.
    """

    @staticmethod
    def serialize_shard_to_file(shard, fp, fs=None, mode: str = "wb", **open_kwargs):
        # Allows to set custom compression
        open_kwargs["mode"] = mode

        if fs is None:
            fs = fsspec

        with fs.open(fp, **open_kwargs) as f:
            for sample in shard:
                f.write(ConfigurableMessagepackSerializer.serialize(sample))

    @staticmethod
    def deserialize_shard_from_file(fp: str, fs=None, mode: str = "rb", max_buffer_size=0, **open_kwargs):
        """
        Reset to max_buffer_size=0 as we trust our dataset source and out-of-buffer errors are not easy do debug for our users.
        """
        open_kwargs["mode"] = mode

        if fs is None:
            fs = fsspec

        with fs.open(fp, **open_kwargs) as f:
            yield from msgpack.Unpacker(f, object_hook=msgpack_numpy.decode, max_buffer_size=max_buffer_size)
