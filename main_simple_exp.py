from main_simple_lib import *

im = load_image('https://previews.123rf.com/images/parilovv/parilovv1804/parilovv180400547/100049238-children-eat-sweets-muffins-cooked-by-mom-did-not-wait-for-milk-concept-is-useful-pastries.jpg')
query = 'How many muffins can each kid have for it to be fair?'

# show_single_image(im)
code = get_code(query)

print("Generated Code: ", code)
