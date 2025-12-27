class MongoQueryBuilder:
    def __init__(self):
        self.query = {}

    def build(self):
        """Возвращает готовый запрос."""
        return self.query

    def add_or(self, *conditions):
        """
        Добавляет $or в запрос.
        :param conditions: Список условий для $or.
        """
        self.query["$or"] = conditions
        return self

    def add_and(self, *conditions):
        """
        Добавляет $and в запрос.
        :param conditions: Список условий для $and.
        """
        self.query["$and"] = conditions
        return self

    def add_condition(self, field, condition):
        """
        Добавляет условие для конкретного поля.
        :param field: Имя поля.
        :param condition: Условие для поля.
        """
        self.query[field] = condition
        return self

    def clear(self):
        """Очищает текущий запрос."""
        self.query = {}
        return self

    @staticmethod
    def build_or(*conditions):
        """
        Создает отдельный $or запрос (для вложений).
        :param conditions: Список условий.
        :return: Словарь с $or.
        """
        return {"$or": conditions}

    @staticmethod
    def build_and(*conditions):
        """
        Создает отдельный $and запрос (для вложений).
        :param conditions: Список условий.
        :return: Словарь с $and.
        """
        return {"$and": conditions}