

class Factory(object):

    @classmethod
    def get_supported_values(cls):
        return {}

    @classmethod
    def factory(cls, cl_type, **kwargs):
        """
        Returns an instance of specified type, built with given, if any, parameters.

        :param cl_type: string name of the classifier class (not case sensitive)
        :param kwargs: additional __init__ parameters
        :return: classifier instance
        """

        key = cl_type.lower()
        supported_values = cls.get_supported_values()
        if supported_values[key]:
            return supported_values[key](**kwargs)
        else:
            raise ValueError('Bad type creation: {}'.format(cl_type))