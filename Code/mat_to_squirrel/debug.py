from mat_to_squirrel.squirrel_ext import ConfigurableMessagePackDriver

path = "/data/datasets/fatih/"
msgpack_driver = ConfigurableMessagePackDriver(path)
it =msgpack_driver.get_iter()
it.take(1).collect()
print("end")
