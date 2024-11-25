from Bio import SeqIO
import os

def pad(segment, gene_length):
    while len(segment) < gene_length:
        segment += segment

    return segment[0:gene_length]

def DNA_dataset_creator(directory, gene_length, method, class_names, class_order, gene_txt, organ_txt, window=False):
    class_name = class_names[class_order]
    gene_file = open(gene_txt, 'a', newline='')
    organ_file = open(organ_txt, 'a', newline='')
    seen = set()
    shorter_contigs = ""
    step = gene_length if not window else window
    


    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        fasta_sequences = SeqIO.parse(open(f),'fasta')

        for fasta in fasta_sequences:
            name, sequence = fasta.id, fasta.seq

            for i in range(0,len(sequence),step):
                seq = sequence[i:i+gene_length]

                if len(seq) < gene_length:
                    if method == 2:
                        shorter_contigs += seq
                    else:
                        seq = pad(seq, gene_length)

                    if window:
                        break

                if seq not in seen:
                    seen.add(seq)
                    gene_file.write(str(seq)+"\n")
                    organ_file.write(str(class_name)+"\n")

            # print (acc,'-------', organism,'------------', tax_line[1],'----------',seq)
            # gene_writer.writerow(seq)
            # organ_writer.writerow(tax_line[1])

    if method == 2:
        for j in range(0,len(shorter_contigs),gene_length):
            contig = shorter_contigs[j:j+gene_length]

            if len(contig) == gene_length:
                if contig not in seen:
                    seen.add(contig)
                    gene_file.write(str(seq)+"\n")
                    organ_file.write(str(class_name)+"\n")

            else:
                last_segment = contig

        last_segment = pad(last_segment, gene_length)

        if last_segment not in seen:
            seen.add(last_segment)
            gene_file.write(str(seq)+"\n")
            organ_file.write(str(class_name)+"\n")


    gene_file.close()
    organ_file.close()
    return [gene_file.name, organ_file.name]
