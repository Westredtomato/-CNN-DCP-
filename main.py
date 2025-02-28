import torch
from PIL import Image, ImageTk
from torchvision import transforms
from haze_removal.haze_removal import remove_haze
from classifier.model import CNNModel  # CNNModel'i import ediyoruz
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# Modeli yükle
model = CNNModel()
model.load_state_dict(torch.load('model.pth'))  # Modeli yüklüyoruz
model.eval()  # Değerlendirme moduna alıyoruz

# Görüntü işleme fonksiyonu
def predict(image_path):
    """
    Bu fonksiyon verilen bir görüntü yoluna göre sisli olup olmadığını sınıflandırır.
    """
    image = Image.open(image_path)  # Görüntüyü açıyoruz
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Görüntüyü yeniden boyutlandırıyoruz
        transforms.ToTensor(),  # Görüntüyü tensöre dönüştürüyoruz
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Görüntüyü normalize ediyoruz
    ])
    image = transform(image).unsqueeze(0)  # Modelin kabul edeceği formata dönüştürüyoruz
    output = model(image)  # Modelden tahmin alıyoruz
    prediction = (output.squeeze() > 0.5).float()  # 0.5 eşik değeri ile sınıflandırıyoruz
    return "Hazy" if prediction.item() == 1 else "Clear"  # Sonucu döndürüyoruz

def process_haze_removal(image_path):
    """
    Sis kaldırma işlemini uygular.
    """
    print("Sis kaldırma işlemi başlıyor...")
    output_image = remove_haze(image_path)  # Sis kaldırma fonksiyonunu çağırıyoruz

    # Convert the output (numpy.ndarray) to a PIL Image
    output_image = Image.fromarray(output_image)  # Convert numpy array to PIL Image
    
    return output_image  # Sis kaldırma sonucu döndürülüyor

def process_image(image_path):
    """
    Görüntüyü sınıflandırır ve gerekirse sisi kaldırır.
    """
    print("Görüntü sınıflandırılıyor...")
    prediction = predict(image_path)  # Görüntüyü sınıflandırıyoruz
    if prediction == "Hazy":
        print("Görüntü sisli, sis kaldırma işlemi başlatılıyor...")
        output_image = process_haze_removal(image_path)  # Sisli ise, sis kaldırma işlemi uygulanır
        return output_image, "Sis kaldırıldı."
    else:
        print("Görüntü sisli değil.")
        return Image.open(image_path), "Görüntü sisli değil."  # Sisli değilse orijinal görüntü döndürülür

# Tkinter GUI kodu
def open_file():
    file_path = filedialog.askopenfilename(title="Bir Görüntü Seçin", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        output_image, message = process_image(file_path)
        messagebox.showinfo("Sonuç", message)
        
        # Sis kaldırılmış görüntü ekrana gösteriliyor
        img = output_image
        img.thumbnail((300, 300))  # Thumbnail boyutu ayarlama
        img = ImageTk.PhotoImage(img)
        result_label.config(image=img)
        result_label.image = img  # To prevent garbage collection

        # "İndir" butonunu aktif hale getir
        download_button.config(state="normal", command=lambda: save_image(output_image))

def save_image(image):
    """
    Kullanıcının sis kaldırılmış görüntüyü kaydetmesi için bir pencere açar.
    """
    output_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")], title="Çıktıyı Kaydet")
    if output_path:  # Eğer kullanıcı bir dosya yolu seçerse
        image.save(output_path)  # Çıktıyı kaydediyoruz
        messagebox.showinfo("Başarılı", f"Dosya {output_path} olarak kaydedildi.")
    else:
        messagebox.showwarning("Uyarı", "Dosya kaydedilmedi.")

# GUI penceresini oluşturuyoruz
root = tk.Tk()
root.title("Sis Kaldırma ve Sınıflandırma")

# Pencereyi güzel bir şekilde konumlandırma
root.geometry("500x600")
root.resizable(False, False)

# Arka plan rengini değiştirme
root.config(bg="#f0f0f0")

# Başlık etiketi
title_label = tk.Label(root, text="Sis Kaldırma Uygulaması", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=20)

# Arayüz bileşenlerini oluşturuyoruz
select_button = tk.Button(root, text="Görüntü Seç", command=open_file, font=("Helvetica", 12), width=20, height=2, bg="#4CAF50", fg="white", relief="solid", bd=2)
select_button.pack(pady=20)

# "İndir" butonu başlangıçta devre dışı
download_button = tk.Button(root, text="İndir", state="disabled", font=("Helvetica", 12), width=20, height=2, bg="#4CAF50", fg="white", relief="solid", bd=2)
download_button.pack(pady=20)

# Sonuç görüntüsü için etiket
result_label = tk.Label(root, bg="#f0f0f0")
result_label.pack(pady=20)

# Alt yazı
footer_label = tk.Label(root, text="Sis tespiti ve kaldırma işlemi", font=("Helvetica", 10), bg="#f0f0f0", fg="#888")
footer_label.pack(side="bottom", pady=10)

# Ana döngüyü başlatıyoruz
root.mainloop()
