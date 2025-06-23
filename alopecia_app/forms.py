from django import forms

class ChatForm(forms.Form):
    prompt = forms.CharField(widget=forms.Textarea, required=False)
    image = forms.ImageField(required=False)  # campo para subir imagen
