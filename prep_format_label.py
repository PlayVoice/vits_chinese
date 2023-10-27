
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter path ...'
    parser.add_argument("--txt", dest="txt")
    parser.add_argument("--out", dest="out")
    args = parser.parse_args()

    ftxt = open(args.txt, "r+", encoding='utf-8')
    scrips = []
    while (True):
        try:
            message = ftxt.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if (message == None):
            break
        if (message == ""):
            break
        infos = message.split('\t', 1)
        word_phone = infos[1].split(' ')
        scrips.append(infos[0][:-4] + ' ' + ''.join(word_phone[0::2]))
        scrips.append('\t' + ' '.join(word_phone[1::2]))

    ftxt.close()

    fout = open(args.out, 'w', encoding='utf-8')
    for item in scrips:
        print(item, file=fout)
    fout.close()
