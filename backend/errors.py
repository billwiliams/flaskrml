from flask import Blueprint, jsonify

errors = Blueprint('errors', __name__)


@errors.app_errorhandler(404)
def not_found(error):
    return (
        jsonify({"success": False,
                "error": 404,
                 "message": "resource not found"}),
        404,
    )


@errors.app_errorhandler(422)
def unprocessable(error):
    return (
        jsonify({"success": False,
                "error": 422,
                 "message": "unprocessable"}),
        422,
    )


@errors.app_errorhandler(400)
def bad_request(error):
    return (jsonify({"success": False,
                    "error": 400,
                     "message": "bad request"}),
            400,
            )


@errors.app_errorhandler(405)
def not_allowed(error):
    return (
        jsonify({"success": False,
                 "error": 405,
                 "message": "method not allowed"}),
        405,

    )
