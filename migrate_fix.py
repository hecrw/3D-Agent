import os
import django
from django.core.management import call_command

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

print("Running makemigrations...")
call_command('makemigrations', 'chat_interface')
print("Running migrate...")
call_command('migrate')
print("Done.")
