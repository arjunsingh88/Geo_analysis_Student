from django.apps import AppConfig


class DashConfig(AppConfig):
    name = 'dash'

    def ready(self):
        import dash.signals
