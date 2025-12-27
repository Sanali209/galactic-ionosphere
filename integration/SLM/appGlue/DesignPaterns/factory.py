class Factory:
    def build(self, **kwargs):
        return None

    def is_supported(self, **kwargs):
        return True


class StaticFactoryRepo:
    factories: dict[str, Factory] = {}

    @staticmethod
    def add_factory(path, factory):
        StaticFactoryRepo.factories[path] = factory

    @staticmethod
    def get_product(path, **kwargs):
        factory = StaticFactoryRepo.factories.get(path)
        if factory is not None:
            return factory.build(**kwargs)
        return None

    def get_factoriesList(self,prefix):
        return [f for f in StaticFactoryRepo.factories if f.startswith(prefix)]


class StaticFactory:
    def build(self, **kwargs):
        return None

    def is_kwargs_valid(self, **kwargs):
        return True

    @staticmethod
    def arg_valid_by_type(prop_name, prop_type: type, default=None, **kwargs):
        arg = kwargs.get(prop_name, default)
        if isinstance(arg, prop_type):
            return arg
        return default
