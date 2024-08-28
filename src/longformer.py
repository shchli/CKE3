
import torch
from transformers import LongformerModel, AutoTokenizer
import argparse



def longforemrss(aaj):
    RELATIONS = {         
    'AtLocation': 'at location',
    'CapableOf': 'capable of',
    'Causes': 'causes',
    'CausesDesire': 'causes desire',
    'DesireOf': 'desire of',
    'Desires': 'desires',
    'HasA': 'has a',
    'HasProperty': 'has property',
    'InheritsFrom': 'inherits from',
    'IsA': 'is a',
    'LocatedNear': 'located near',
    'LocationOfAction': 'location of action',
    'MotivatedByGoal': 'motivated by goal',
    'NotHasA': 'not has a',
    'NotHasProperty': 'not has property',
    'NotIsA': 'not is a',
    'PartOf': 'part of',
    'ReceivesAction': 'receives action',
    'RelatedTo': 'related to',
    'SymbolOf': 'symbol of'
    }

    model = LongformerModel.from_pretrained("/home/icdm/NewWorld/lsc/TiRGN-main/recformer/longformer-base-4096")
    tokenizer = AutoTokenizer.from_pretrained("/home/icdm/NewWorld/lsc/TiRGN-main/recformer/longformer-base-4096")
    model.to('cuda:0')

    datapath = '/home/icdm/NewWorld/lsc/TiRGN-main/con_models/conceptnet/data/all_e_YAGO.pickle'
    with open(datapath,'rb') as df:

        output_list = []
        all_knowledge = torch.load(df, map_location='cpu')
    i = 0
    numbers = aaj * 20
    if (numbers+20)>len(all_knowledge):
      all_knowledgel = all_knowledge[numbers:len(all_knowledge)]
    else:  
        all_knowledgel = all_knowledge[numbers:numbers+20]
    for knowledgetext in all_knowledgel:
        knowledge_list = []
        for text in knowledgetext:
        
            if len(knowledge_list)==0:
                if len(text)==3:
                    knowledge_list.append('[CLS]'+ ' ' + text['e1']+ ' ' + RELATIONS[text['relation']] + ' ' + text['beams'])
                else:
                    knowledge_list.append('[CLS]'+ ' ' + text['e1'])
                    break
            else:

                if len(text)==3:
                    knowledge_list.append(' ' + RELATIONS[text['relation']] + ' ' + text['beams'])
                # else:
                #     knowledge_list.append(text['e1']+ ' ' + RELATIONS[text['relation']])

        SAMPLE_TEXT = "".join(knowledge_list)  # long input document
        input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)  # initialize to local attention
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)  # initialize to global attention to be deactivated for all tokens
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        global_attention_mask = global_attention_mask.cuda()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)[0].squeeze(0)
        outputs = outputs.cpu()
        output = outputs[0]
        output_list.append(output)
        print(i)
        i += 1
        if i == 20:
            break
    # outputtensor = torch.stack(output_list, dim=0)
    # outputtensor1 = torch.narrow  (outputtensor,0, 0,20)
    # outputtensor2 = outputtensor.repeat (47, 1)
    # outputtensor0 = torch.cat([outputtensor1,outputtensor2],dim=0)
    aai = str(args.i)
    torch.save(output_list,"/home/icdm/NewWorld/lsc/TiRGN-main/recformer/data/temp/temp_e_YAGO_rec_" + aai +".pickle")
    
def outputtensor():
    t_l = []
    for ii in range(532):
        trmpl = torch.load("/home/icdm/NewWorld/lsc/TiRGN-main/recformer/data/temp/temp_e_YAGO_rec_" + str(ii) +".pickle")
        t_l = t_l +trmpl
    outputtensor = torch.stack(t_l, dim=0)
    torch.save(outputtensor,"/home/icdm/NewWorld/lsc/TiRGN-main/recformer/data/all_e_YAGO_rec.pickle")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, default=0)
    args = parser.parse_args()
    longforemrss(args.i)
    # outputtensor()
