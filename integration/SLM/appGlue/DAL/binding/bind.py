from SLM.appGlue.DAL.DAL import ForwardConverter, DataConverterFactory, GlueDataConverter
from SLM.appGlue.DesignPaterns.factory import StaticFactory


class damValidator:
    def validate(self, value):
        return True


class BindingObserver:
    def __init__(self):
        self.source_property: BindingProperty = None
        self.converter = ForwardConverter()
        self.validator = damValidator()
        self.child_obs = []
        self.callback = None

    def on_registered(self):
        prop_val = self.converter.Convert(self.source_property.val)
        if self.callback is not None:
            self.callback(prop_val)

    def on_prop_changed(self, prop, val):
        pass

    def prop_changed(self, prop):
        prop_val = self.converter.Convert(prop.val)
        self.on_prop_changed(prop, prop_val)
        if self.callback is not None:
            self.callback(prop_val)

    def set_prop_val(self, value):
        converted_val = self.converter.ConvertBack(value)
        self.source_property.set_value(converted_val, self)
        if self.callback is not None:
            self.callback(converted_val)

    def get_prop_val(self):
        return self.converter.Convert(self.source_property.val)


class PropBinding(BindingObserver):
    def __init__(self, target_prop, converter=None, one_way=False):
        super().__init__()
        self.target_property = target_prop
        if not one_way:
            self.listener = self.target_changed
            self.target_property.add_listener(self.listener)
        if converter is not None:
            self.converter = converter

    def target_changed(self, value):
        self.set_prop_val(value)

    def on_registered(self):
        self.target_property.val = self.get_prop_val()

    def on_prop_changed(self, prop, val):
        self.target_property.val = self.get_prop_val()


class PropUser:
    def __init__(self):
        self.props = {}
        for key, val in self.__class__.__dict__.items():
            if val is not None and isinstance(val, PropInfo):
                val.prop_name = key
                if not val.persist:
                    self.props[key] = BindingProperty(val.defaulth)
                else:
                    self.props[key] = BindingProperty(val.defaulth)
        self.dispatcher = PropDispatcher(self)


class PropInfo:
    def __init__(self, default=None, persist=False):
        self.defaulth = default
        self.prop_name = ""
        self.persist = persist

    def __get__(self, instance, owner):
        return instance.props[self.prop_name].val

    def __set__(self, instance, value):
        instance.props[self.prop_name].val = value


class BindingProperty:
    def __init__(self, default=None):
        self.observers = []
        self._cached_value = default
        self.call_backs = []

    def register(self, observer):
        observer.source_property = self
        self.observers.append(observer)
        observer.on_registered()

    def bind(self, bind_target=None, **kwargs):
        if bind_target is None:
            return
        kwargs["bind_target"] = bind_target
        bind_fact = BindFactoryBase()
        if bind_fact.is_supported(**kwargs):
            obs: BindingObserver = bind_fact.build(**kwargs)
            self.register(obs)

    def fire_prop_changed(self, sender=None):
        for obs in self.observers:
            if obs != sender:
                obs.prop_changed(self)
        for cb in self.call_backs:
            cb(self.val)

    @property
    def val(self):
        return self._cached_value

    @val.setter
    def val(self, value):
        self.set_value(value)

    def set_value(self, value, sender=None):
        self._cached_value = value
        self.fire_prop_changed(sender)

    def add_listener(self, delegate):
        self.call_backs.append(delegate)


class PropDispatcher:
    def __init__(self, disp):
        self.disp = disp

    def __getattr__(self, item) -> BindingProperty:
        obj_property = self.disp.props.get(item)
        if obj_property is None:
            raise AttributeError(f"property {item} not found")
        return obj_property


class BindFactoryBase(StaticFactory):
    bind_dict = {BindingProperty: PropBinding}

    def build(self, **kwargs):
        kwargs = self.transform_params(**kwargs)
        bind_obs = self.get_obs(**kwargs)
        return bind_obs

    def get_bind_target(self, **kwargs):
        return self.arg_valid_by_type("bind_target", object, **kwargs)

    def get_obs(self, bind_target, **kwargs):
        t = type(bind_target)
        kwargs["bind_target"] = bind_target
        obs = self.bind_dict[t](**kwargs)
        return obs

    def transform_params(self, **kwargs):
        # move parameter conversation to prop constructor
        converter_name = self.arg_valid_by_type("converter",
                                                str,
                                                default=None, **kwargs)
        converter_inst = DataConverterFactory().build(converter_name=converter_name)
        if converter_inst is not None:
            kwargs["converter"] = converter_inst
        return kwargs

    def is_supported(self, **kwargs):
        bind_targ: object = kwargs.get("bind_target")
        t = type(bind_targ)
        if t in self.bind_dict:
            return True
