from data.dataset import SpamDataset
from training.trainer import BertTrainer
from config import settings
from models.bert_model import BertModel
from utils import cleaner
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def train(model_path):
    texts = [
    "Takipçi kazanmak isteyenler DM atsın",
    "Youtube kanalım için abone olun lütfen",
    "Merhaba, nasılsınız?",
    "Bugün hava gerçekten çok güzel",
    "Telegram grubumuza katılmak ister misiniz?",
    "Okuldan çıktım, acayip yorgunum",
    "Hemen şimdi tıklayın ve kazanmaya başlayın",
    "Öğle yemeğinde ne yesek acaba?",
    "Kanalıma abone olursanız sevinirim",
    "Film öneriniz var mı bu akşam?",
    "Kazanmak için bu bağlantıya tıklayın",
    "Bugün toplantı saat kaçta başlıyor?",
    "Son 5 kişiye hediye gönderiyoruz",
    "Çalışma saatleriniz nedir acaba?",
    "Bu fırsat kaçmaz!",
    "Dostlarla kahve içmek gibisi yok",
    "Profilimi takip eden herkes kazanıyor",
    "Sinemaya gidiyoruz, gelmek ister misin?",
    "Bugün özel indirim günü",
    "İyi akşamlar, görüşmek üzere",
    "Hemen şimdi sipariş ver",
    "Bu sabah sınav vardı, zor geçti",
    "Takip edeni takip ederim",
    "Saat kaçta buluşuyoruz",
    "Ücretsiz çekiliş için yorum yapın",
    "Yeni kitap aldım, harika görünüyor",
    "Linke tıklayan ilk 100 kişiye ödül",
    "Akşam için yemek önerisi var mı?",
    "Hediye kazanmak için hemen kaydolun",
    "Bugün çok güzel bir yürüyüş yaptım"
    ]   

    labels = [
        1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    tokenizer=AutoTokenizer.from_pretrained(settings.MODEL_NAME)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
    )

    train_dataset = SpamDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer)
    eval_dataset = SpamDataset(texts=eval_texts, labels=eval_labels, tokenizer=tokenizer)
    model=BertModel()
    model.load_pretrained(model_path)
    trainer=BertTrainer(model.model, model.tokenizer)
    trainer.train(train_dataset, eval_dataset)
def main(model_instance=None):
    model=BertModel().load_pretrained(model_instance)
    while True:
        text=input("Type a text: ")
        textfix=cleaner.preprocess_text(text=text)
        prediction=model.predict(text=textfix)
        if prediction[0]==0:
            print(text)
            print("it is spam!")
        elif prediction[0]==1:
            print(text)
            print("it is not spam!")

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--model_path", type=str, default=settings.MODEL_DIR)
    args=parser.parse_args()
    if args.train:
        train(args.model_path)
    elif args.start:
        main(args.model_path/"checkpoint-6")