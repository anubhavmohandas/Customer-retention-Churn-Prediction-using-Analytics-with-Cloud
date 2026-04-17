from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('customer', '0006_loginhistory'),
    ]

    operations = [
        migrations.CreateModel(
            name='ActivityLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('action', models.CharField(
                    choices=[
                        ('LOGIN_FAILED', 'Failed Login Attempt'),
                        ('LOGOUT',       'Logout'),
                        ('BULK_PREDICT', 'Bulk Prediction Run'),
                        ('CSV_EXPORT',   'CSV Export / Download'),
                    ],
                    db_index=True,
                    max_length=20,
                )),
                ('ip_address', models.GenericIPAddressField(blank=True, null=True)),
                ('detail', models.CharField(blank=True, max_length=500)),
                ('attempted_email', models.EmailField(blank=True)),
                ('user', models.ForeignKey(
                    blank=True,
                    null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='activity_log',
                    to=settings.AUTH_USER_MODEL,
                )),
            ],
            options={
                'verbose_name': 'Activity Log',
                'verbose_name_plural': 'Activity Log',
                'ordering': ['-timestamp'],
            },
        ),
    ]
