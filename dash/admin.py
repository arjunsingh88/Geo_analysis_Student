from import_export import resources, fields, widgets
from import_export.admin import ImportExportModelAdmin
from django.contrib import admin
from .models import Profile, Student, Internship, College, Country


class ProfileAdmin(admin.ModelAdmin):
    list_display = {'title', 'date_created', 'last_modified', 'is_draft'}


admin.site.register(Profile)


class StudentResource(resources.ModelResource):
    delete = fields.Field(widget=widgets.BooleanWidget())

    def for_delete(self, row, instance):
        return self.fields['delete'].clean(row)
    class Meta:
        model = Student


@admin.register(Student)
class StudentAdmin(ImportExportModelAdmin,):
    pass


class CollegeResource(resources.ModelResource):
    class Meta:
        model = College


@admin.register(College)
class CollegeAdmin(ImportExportModelAdmin,):
    pass


class InternshipResource(resources.ModelResource):
    class Meta:
        model = Internship


@admin.register(Internship)
class InternshipAdmin(ImportExportModelAdmin,):
    pass


class CountryResource(resources.ModelResource):
    class Meta:
        model = Country


@admin.register(Country)
class CountryAdmin(ImportExportModelAdmin,):
    pass


