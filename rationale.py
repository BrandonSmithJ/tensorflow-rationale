from IPython.core.display import display, HTML, clear_output
import ipywidgets as widgets
import json, os

ASPECT = 'Taste'

def color(word, prob, c):
    colors = {'red'     : '255,0,0,%s',
              'green'   : '0,255,0,%s',
              'blue'    : '0,0,255,%s',
              'orange'  : '255,165,0,%s',
              'yellow'  : '250,218,94,%s',
    }
    c = colors[c] % prob
    ele = '<span style="background:rgba(%s)">%s</span>'
    return ele % (c, word), prob


def set_text(attribute, text, probs):
    colors = {'Appearance' : 'blue',
              'Aroma'      : 'green',
              'Palate'     : 'orange',
              'Taste'      : 'red',
              'Overall'    : 'yellow'}
    c = colors[attribute]
    html = [color(word, probs[i], c) for i,word in enumerate(text.split())]
    html = [[span, color(' ', abs(prob+html[i+1][1])/2., c)[0]] for i, (span, prob) in enumerate(html[:-1])] + [html[-1][0]]
    html = [w2 for w1 in html for w2 in w1]
    html = ''.join(html)
    return html


class Aspect(object):

    def load_aspect(self, aspect):
        filename = '%s.json' % aspect 
        if not os.path.exists(filename):
            raise Exception('Aspect "%s" does not exist - need to train model on this aspect and output estimates' % aspect)

        outputs = [json.loads(line) for line in open(filename).readlines()]

        text_list = [' '.join([ch for ch in o['original'].split() if ch != '_']) for o in outputs]
        prob_list = [[float(p) for p in eval(o['prob']) if float(p) > 0.] for o in outputs]
        targ_list = [[float(t) for t in eval(o['y'])] for o in outputs]
        pred_list = [[float(p) for p in eval(o['p'])] for o in outputs]

        c = 1e-1
        for j,prob in enumerate(prob_list):
            for i,p in enumerate(prob[2:-2],2):
                if p <= c and prob[i+1] > c and prob[i-1] > c and (prob[i+2] > c or prob[i-2] > c):
                    prob[i] = (prob[i-1] + prob[i+1]) / 2
                elif p > c and prob[i-1] <= c and prob[i+1] <= c and (prob[i+2] <= c or prob[i-2] <= c):
                    prob[i] = 0
            prob_list[j] = prob


        assert(all([len(p)==len(t.split()) for t,p in zip(text_list, prob_list)])), [len(text_list[0]), len(prob_list[0])]
    
        self.text_list = text_list
        self.prob_list = prob_list
        self.targ_list = targ_list
        self.pred_list = pred_list
        self.prob_list = prob_list
        return self


data = Aspect().load_aspect(ASPECT)

target  = widgets.Label('Target: %s'%data.targ_list[0][3], layout=widgets.Layout(margin='0px 10px 0px 0px'))
predict = widgets.Label('Prediction: %.2f'%data.pred_list[0][3])
stats   = widgets.HBox([target, predict])
attrs   = ['Appearance', 'Aroma', 'Palate', 'Taste', 'Overall']

class TextController(object):
    def __init__(self, dropdown, button, stats, data):
        self.idx  = 0
        self.attr = ASPECT
        self.dropdown = dropdown
        self.button = button 
        self.stats = stats
        self.data = data
        self.text = ''

    def drop_select(self, b):
        if b['name'] == 'value':
            self.idx = idx = b['new']
            self.radio_select({'name':'value', 'new':self.attr})

    def radio_select(self, b):
        if b['name'] == 'value':
            if os.path.exists('%s.json' % b['new']):
                if b['new'] != self.attr:
                    self.attr = b['new']
                    self.data.load_aspect(self.attr)
                self.set_text()
            else:
                self.attr = b['new']
                self.text = 'Aspect "%s" does not exist - need to train model on this aspect and output estimates' % b['new']
                self.show()

    def set_text(self):
        self.text = HTML(set_text(self.attr, self.data.text_list[self.idx], self.data.prob_list[self.idx]))
        target.value = 'Target: %s' % self.data.targ_list[self.idx][attrs.index(self.attr)]
        predict.value = 'Prediction: %.2f' % self.data.pred_list[self.idx][attrs.index(self.attr)]
        self.show()

    def show(self):
        clear_output()
        display(self.dropdown)
        display(self.button)
        display(self.stats)        
        display(self.text)


dropdown = widgets.Dropdown(
    options=list(range(len(data.text_list))),
    value=0,
    description='Sample:',
)
button = widgets.RadioButtons(
    options=attrs,
    description='Attribute:',
    disabled=False
)    


T = TextController(dropdown, button, stats, data)

dropdown.observe(T.drop_select)
button.observe(T.radio_select)

button.value = ASPECT
button.selected_label = ASPECT
T.set_text()