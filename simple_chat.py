import ccm as ccm

ccm_chat = ccm.CCM(file="RatStupid_18M.json")
while True:
    input_user = input("You: ").lower()
    answer = ccm_chat.chat([{"role": "user", "content": input_user}])
    print("CCM:", answer)