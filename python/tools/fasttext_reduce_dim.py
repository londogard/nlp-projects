import fasttext
import fasttext.util

ft = fasttext.load_model('cc.sv.300.bin')
print(ft.get_dimension())

fasttext.util.reduce_model(ft, 50)
print(ft.get_dimension())
ft.save_model('cc.sv.50.bin')