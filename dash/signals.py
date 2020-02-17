from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from import_export.signals import post_import, post_export
from . models import Profile


@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_profile(sender, instance, **kwargs):
    instance.profile.save()


@receiver(post_import, dispatch_uid='deleted')
def _post_import(model, **kwargs):
    # model is the actual model instance which after import
    pass


@receiver(post_export, dispatch_uid='deleted')
def _post_export(model, **kwargs):
    # model is the actual model instance which after export
    pass

