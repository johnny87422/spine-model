def context_wrapper(status, message='', data=None, errors=None, **kwargs):

    """A wrapper for api response format

    api format:
    {
        'status': status,
        'message': message,
        'data': data
    }
    """

    status_map = {
        200: 'Success',
        400: 'Invalid Parameters',
        404: 'Not Found',
        500: 'Internal Error'
    }

    wrapper = {
        'status': status,
        'message': message or status_map.get(status, message),
        'data': data if data is not None else {}
    }

    if errors is not None:
        wrapper['errors'] = errors

    wrapper.update(**kwargs)

    return wrapper