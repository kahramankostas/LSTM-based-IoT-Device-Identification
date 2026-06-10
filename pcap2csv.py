#!/usr/bin/env python
# coding: utf-8

# In[10]:


#!/usr/bin/env python
# coding: utf-8

import os
import math
import csv
import warnings
import pandas as pd
from tqdm import tqdm
from scapy.all import rdpcap, IP, TCP, UDP, ICMP, ARP, LLC, EAPOL, Raw, IPOption_Router_Alert

warnings.filterwarnings("ignore")

# ==========================================
# 1. YARDIMCI FONKSİYONLAR
# ==========================================

def find_the_way(path, file_format):
    """Belirtilen dizindeki belirli formattaki dosyaların yollarını bulur."""
    files_add = []
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add

def create_folder(f_name):
    """Klasör yoksa oluşturur."""
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print(f"Hata: {f_name} klasörü oluşturulamadı!")

def port_class(port):
    """Port numaralarını sınıflandırır."""
    if 0 <= port <= 1023:
        return 1
    elif 1024 <= port <= 49151:
        return 2
    elif 49152 <= port <= 65535:
        return 3
    else:
        return 0

def shannon(data):
    """Verilen bayt dizisi için Shannon Entropisini hesaplar."""
    if not data:
        return 0.0

    LOG_BASE = 2
    dataSize = len(data)
    ent = 0.0
    freq = {} 

    for c in data:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1

    for key in freq.keys():
        f = float(freq[key]) / dataSize
        if f > 0:
            ent = ent + f * math.log(f, LOG_BASE)
    return -ent

def pre_entropy(payload):
    """Payload verisini entropi hesabına hazırlar."""
    characters = [i for i in payload]
    return shannon(characters)


# In[11]:


# ==========================================
# 2. SABİTLER VE LİSTELER
# ==========================================

MAC_list = {
    '00:17:88:24:76:ff': 'Hue-Device',
    '00:1a:22:03:cb:be': 'MAXGateway',
    '00:1a:22:05:c4:2e': 'HomeMaticPlug',
    '00:24:e4:24:80:2a': 'Withings',
    '00:b5:6d:06:08:ba': 'unknown',
    '1c:5f:2b:aa:fd:4e': 'D-LinkDevice',
    '20:f8:5e:ca:91:52': 'Aria',
    '24:77:03:7c:ea:dc': 'unknown',
    '28:b2:bd:c3:41:79': 'unknown',
    '38:0b:40:ef:85:41': 'unknown',
    '50:c7:bf:00:c7:03': 'TP-LinkPlugHS110',
    '50:c7:bf:00:fc:a3': 'TP-LinkPlugHS100',
    '3c:49:37:03:17:db': 'EdnetCam',
    '3c:49:37:03:17:f0': 'EdnetCam',
    '5c:cf:7f:06:d9:02': 'iKettle2',
    '5c:cf:7f:07:ae:fb': 'SmarterCoffee',
    '6c:72:20:c5:17:5a': 'D-LinkWaterSensor',
    '74:da:38:23:22:7b': 'EdimaxPlug2101W',
    '74:da:38:4a:76:49': 'EdimaxPlug1101W',
    '74:da:38:80:79:fc': 'EdimaxCam',
    '74:da:38:80:7a:08': 'EdimaxCam',
    '84:18:26:7b:5f:6b': 'Lightify',
    '90:8d:78:a8:e1:43': 'D-LinkSensor',
    '90:8d:78:a9:3d:6f': 'D-LinkSwitch',
    '90:8d:78:dd:0d:60': 'D-LinkSiren',
    '94:10:3e:34:0c:b5': 'WeMoSwitch',
    '94:10:3e:35:01:c1': 'WeMoSwitch',
    '94:10:3e:41:c2:05': 'WeMoInsightSwitch',
    '94:10:3e:42:80:69': 'WeMoInsightSwitch',
    '94:10:3e:cd:37:65': 'WeMoLink',
    'ac:cf:23:62:3c:6e': 'EdnetGateway',
    'b0:c5:54:1c:71:85': 'D-LinkDayCam',
    'b0:c5:54:25:5b:0e': 'D-LinkCam',
    'bc:f5:ac:f4:c0:9d': 'unknown'
}


# ==========================================
# 3. VERİ ÇIKARIMI (FEATURE EXTRACTION)
# ==========================================

folder_name = "./csvs/"
create_folder(folder_name)

files_add = find_the_way('./captures_IoT-Sentinel/', '.pcap')


# In[12]:


for file_path in tqdm(files_add):
    pkt = rdpcap(file_path)
    print(f"\n===================== {os.path.basename(file_path)} =====================")
    print(pkt)

    csvname = file_path.replace(".pcap", ".csv")
    ip_add_count = 0
    dst_ip_list = []

    with open(csvname, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        for p in pkt:
            # Başlangıç değerleri
            layer_2_arp, layer_2_llc = 0, 0
            layer_3_eapol, layer_3_ip, layer_3_icmp, layer_3_icmp6 = 0, 0, 0, 0
            layer_4_tcp, layer_4_udp, layer_4_tcp_ws = 0, 0, 0
            layer_7_http, layer_7_https, layer_7_dhcp, layer_7_bootp = 0, 0, 0, 0
            layer_7_ssdp, layer_7_dns, layer_7_mdns, layer_7_ntp = 0, 0, 0, 0
            ip_padding, ip_ralert = 0, 0
            port_class_src, port_class_dst = 0, 0
            pck_size, pck_rawdata, entropy = 0, 0, 0

            # Paket Boyutu
            pck_size = len(p)

            # Layer 2
            if p.haslayer(ARP): layer_2_arp = 1
            if p.haslayer(LLC): layer_2_llc = 1

            # Layer 3
            if p.haslayer(EAPOL): layer_3_eapol = 1
            if p.haslayer(ICMP): layer_3_icmp = 1

            if p.haslayer(IP):
                layer_3_ip = 1
                if hasattr(p[IP], 'version') and p[IP].version == 6:  # IPv6 Tespiti
                    layer_3_icmp6 = 1

                # IP Çeşitliliği Takibi
                dst_ip = p[IP].dst
                if dst_ip not in dst_ip_list:
                    ip_add_count += 1
                    dst_ip_list.append(dst_ip)

                # Port Sınıflandırması
                if hasattr(p[IP], 'sport'): port_class_src = port_class(p[IP].sport)
                if hasattr(p[IP], 'dport'): port_class_dst = port_class(p[IP].dport)

                # IP Seçenekleri (Options)
                if p[IP].ihl > 5:
                    if p.haslayer(IPOption_Router_Alert):
                        ip_ralert = 1
                        if "Padding" in repr(p[IPOption_Router_Alert]):
                            ip_padding = 1

            # Layer 4 - UDP
            if p.haslayer(UDP):
                layer_4_udp = 1
                sport, dport = p[UDP].sport, p[UDP].dport
                if 67 in (sport, dport) or 68 in (sport, dport):
                    layer_7_dhcp = 1
                    layer_7_bootp = 1
                if 53 in (sport, dport): layer_7_dns = 1
                if 5353 in (sport, dport): layer_7_mdns = 1
                if 1900 in (sport, dport): layer_7_ssdp = 1
                if 123 in (sport, dport): layer_7_ntp = 1

            # Layer 4 - TCP
            if p.haslayer(TCP):
                layer_4_tcp = 1
                layer_4_tcp_ws = p[TCP].window
                sport, dport = p[TCP].sport, p[TCP].dport
                if 80 in (sport, dport): layer_7_http = 1
                if 443 in (sport, dport): layer_7_https = 1

            # Payload Analizi (Entropy & Raw Data)
            if p.haslayer(Raw):
                pck_rawdata = 1
                entropy = pre_entropy(p[Raw].original)

            # Etiketleme Mantığı
            Mac = getattr(p, 'src', "")
            if Mac in ["1c:5f:2b:aa:fd:4e", "00:17:88:24:76:ff"]:
                path_parts = os.path.normpath(file_path).split(os.sep)
                label = path_parts[1] if len(path_parts) > 1 else Mac
            else:
                label = MAC_list.get(Mac, Mac)

            # Satırı Hazırlama ve Yazma
            line = [
                layer_2_arp, layer_2_llc, layer_3_eapol, layer_3_ip, layer_3_icmp, layer_3_icmp6, 
                layer_4_tcp, layer_4_udp, layer_4_tcp_ws, layer_7_http, layer_7_https, layer_7_dhcp, 
                layer_7_bootp, layer_7_ssdp, layer_7_dns, layer_7_mdns, layer_7_ntp, ip_padding, 
                ip_add_count, ip_ralert, port_class_src, port_class_dst, pck_size, pck_rawdata, 
                entropy, label
            ]
            csv_writer.writerow(line)

print("Pcap analizleri tamamlandı. Veri setleri birleştiriliyor...\n")




# In[13]:


# ==========================================
# 4. VERİ SETLERİNİ BİRLEŞTİRME (TRAIN/TEST/VALIDATION)
# ==========================================

# Not: Veri listeleriniz okunaklı olması amacıyla sıkıştırılmıştır.
validation=[ 'captures_IoT-Sentinel/Aria/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-2-STA.csv', 
            'captures_IoT-Sentinel/EdnetCam1/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/EdnetCam2/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/EdnetCam2/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/EdnetCam2/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/EdnetCam2/Setup-C-3-STA.csv',
            'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-16-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-17-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam2/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam2/Setup-A-2-STA.csv', 
            'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam2/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-4-STA.csv', 
            'captures_IoT-Sentinel/Withings/Setup-C-18-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-19-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-4-STA.csv', 
            'captures_IoT-Sentinel/WeMoSwitch/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch2/Setup-C-1-STA.csv', 
            'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-2-STA.csv',
            'captures_IoT-Sentinel/iKettle2/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-6-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-7-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-8-STA.csv',
            'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-19-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-20-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-3-STA.csv',
            'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-12-STA.csv', 
            'captures_IoT-Sentinel/D-LinkSiren/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-16-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-17-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-9-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-4-STA.csv', 
            'captures_IoT-Sentinel/Lightify/Setup-C-19-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-20-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-3-STA.csv',
            'captures_IoT-Sentinel/HueBridge/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-4-STA.csv',
            'captures_IoT-Sentinel/EdnetGateway/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-16-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-19-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-20-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch2/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch2/Setup-C-3-STA.csv',
            'captures_IoT-Sentinel/D-LinkCam/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-C-2-STA.csv',
            'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-C-2-STA.csv', 
            'captures_IoT-Sentinel/SmarterCoffee/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-6-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-7-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-8-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-9-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-C-4-STA.csv',
            'captures_IoT-Sentinel/WeMoLink/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-6-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-7-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-8-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-9-STA.csv', 
            'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-C-9-STA.csv',
            'captures_IoT-Sentinel/MAXGateway/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch2/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch2/Setup-C-2-STA.csv']


test=[ 'captures_IoT-Sentinel/Aria/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-9-STA.csv',
      'captures_IoT-Sentinel/Withings/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-9-STA.csv', 
      'captures_IoT-Sentinel/WeMoSwitch/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-C-6-STA.csv', 
 'captures_IoT-Sentinel/D-LinkCam/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch2/Setup-C-4-STA.csv', 
      'captures_IoT-Sentinel/EdnetCam2/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/EdnetCam2/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/EdnetCam2/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/EdnetCam2/Setup-C-7-STA.csv', 
      'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/EdnetCam2/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-C-3-STA.csv',
      'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch2/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch2/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch2/Setup-C-5-STA.csv', 
 'captures_IoT-Sentinel/iKettle2/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-C-4-STA.csv',
      'captures_IoT-Sentinel/D-LinkDayCam/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-6-STA.csv', 
      'captures_IoT-Sentinel/D-LinkHomeHub/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-C-5-STA.csv', 
      'captures_IoT-Sentinel/D-LinkSiren/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-C-9-STA.csv', 
      'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-C-7-STA.csv', 
      'captures_IoT-Sentinel/EdnetGateway/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-9-STA.csv',
      'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam2/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam2/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam2/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam2/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam2/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-C-5-STA.csv', 
      'captures_IoT-Sentinel/Lightify/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-9-STA.csv', 
      'captures_IoT-Sentinel/HueBridge/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-A-5-STA.csv',
'captures_IoT-Sentinel/HueSwitch/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch2/Setup-C-5-STA.csv']


train=['captures_IoT-Sentinel/Aria/Setup-A-1-STA.csv',
       'captures_IoT-Sentinel/WeMoLink/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-10-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-11-STA.csv',
       'captures_IoT-Sentinel/Lightify/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-12-STA.csv',
       'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-4-STA.csv',
       'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-4-STA.csv',
       'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug1101W/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/EdimaxPlug2101W/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/D-LinkHomeHub/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-A-1-STA.csv',
       'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-18-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-19-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/D-LinkWaterSensor/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-A-6-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-A-7-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/EdimaxCam1/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-B-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-B-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-B-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-B-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkSensor/Setup-B-5-STA.csv',
       'captures_IoT-Sentinel/D-LinkSiren/Setup-C-18-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-19-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/D-LinkSiren/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/D-LinkSwitch/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/D-LinkDayCam/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-16-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-17-STA.csv',
 'captures_IoT-Sentinel/D-LinkDoorSensor/Setup-C-18-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-14-STA.csv',
       'captures_IoT-Sentinel/EdnetGateway/Setup-C-17-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-18-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-3-STA.csv',
       'captures_IoT-Sentinel/EdnetCam1/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/EdnetCam1/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/EdnetCam1/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/EdnetCam1/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/EdnetCam1/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/EdnetCam1/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/EdnetCam1/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/EdnetCam1/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/EdnetCam1/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/EdnetCam1/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/EdnetGateway/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-16-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-17-STA.csv',
 'captures_IoT-Sentinel/HomeMaticPlug/Setup-C-18-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-16-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-17-STA.csv', 
       'captures_IoT-Sentinel/iKettle2/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-10-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-11-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-12-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-13-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-14-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-15-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-2-STA.csv',
       'captures_IoT-Sentinel/HueBridge/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-3-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-4-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-5-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-6-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-7-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-8-STA.csv',
 'captures_IoT-Sentinel/HueBridge/Setup-C-9-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-A-10-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-A-6-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-A-7-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-A-8-STA.csv',
 'captures_IoT-Sentinel/HueSwitch/Setup-A-9-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/iKettle2/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/Lightify/Setup-C-18-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-12-STA.csv',
       'captures_IoT-Sentinel/SmarterCoffee/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-10-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-11-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-12-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-13-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-14-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-15-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/SmarterCoffee/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-13-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-14-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-15-STA.csv',
       'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS100/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/WeMoLink/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-A-3-STA.csv', 
       'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-A-10-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-A-6-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-A-7-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-A-8-STA.csv',
 'captures_IoT-Sentinel/TP-LinkPlugHS110/Setup-A-9-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-1-STA.csv', 
       'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-A-10-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-A-11-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-A-6-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-A-7-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-A-8-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-A-9-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/WeMoInsightSwitch/Setup-C-2-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/Aria/Setup-C-13-STA.csv',
       'captures_IoT-Sentinel/Withings/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-11-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-12-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-13-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-14-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-15-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-16-STA.csv', 
       'captures_IoT-Sentinel/MAXGateway/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-A-6-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-A-7-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-A-8-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-A-9-STA.csv',
 'captures_IoT-Sentinel/MAXGateway/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/Withings/Setup-C-17-STA.csv', 
       'captures_IoT-Sentinel/WeMoSwitch/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-A-10-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-A-6-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-A-7-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-A-8-STA.csv', 
       'captures_IoT-Sentinel/D-LinkCam/Setup-A-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-A-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-A-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-A-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-A-5-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-B-1-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-B-2-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-B-3-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-B-4-STA.csv',
 'captures_IoT-Sentinel/D-LinkCam/Setup-B-5-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-A-9-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-C-1-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-C-10-STA.csv',
 'captures_IoT-Sentinel/WeMoSwitch/Setup-C-2-STA.csv']


# In[14]:


df_list = {
    "Train_IoTDevIDv1.csv": train,
    "Validation_IoTDevIDv1.csv": validation,
    "Test_IoTDevIDv1.csv": test
}

col_names = [
    'ARP', 'LLC', 'EAPOL', "IP", 'ICMP', 'ICMP6', 'TCP', 'UDP', 'TCP_w_size', 'HTTP', 
    'HTTPS', 'DHCP', 'BOOTP', 'SSDP', 'DNS', 'MDNS', 'NTP', 'IP_padding', 'IP_add_count', 
    'IP_ralert', 'Portcl_src', 'Portcl_dst', 'Pck_size', 'Pck_rawdata', "Entropy", 'Label'
]

# Pandas ile verimli birleştirme
for output_name, file_list in df_list.items():
    print(f"Oluşturuluyor: {output_name}")

    # Mevcut ve başarılı CSV'leri listeye toplayın
    dataframes = []
    for file in tqdm(file_list):
        if os.path.exists(file):
            try:
                # Orijinal dosyalar header olmadan yazıldığı için 'names' kullanıyoruz
                temp_df = pd.read_csv(file, names=col_names)
                dataframes.append(temp_df)
            except Exception as e:
                print(f"Hata Okunurken: {file} - {e}")

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        # 'unknown' etiketine sahip olanları filtreleyin
        combined_df = combined_df[combined_df["Label"] != "unknown"]
        # Sonucu CSV'ye aktarın
        combined_df.to_csv(output_name, index=False)
        print(f"\n{output_name} dağılımı:")
        print(combined_df.groupby("Label").size(), "\n")
    else:
        print(f"Uyarı: {output_name} için veri bulunamadı.\n")


# In[ ]:





# In[ ]:




