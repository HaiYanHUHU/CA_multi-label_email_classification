import pandas as pd
import typing as t
import re
from config import Config

def get_input_data(filename: t.Union[t.Sequence[str], str]) -> pd.DataFrame:
    """
    Loads and preprocesses csvs in the format established by the company
    :param filename: path(s) to csv file
    :return: pd.DataFrame
    """
    df = pd.concat([pd.read_csv(p) for p in ((filename,) if isinstance(filename, str) else filename)])

    # Drops row if TICKET_SUMMARY_COL and/or INTERACTION_CONTENT_COL has na values
    df = df[
        df[[Config.TICKET_SUMMARY_COL, Config.INTERACTION_CONTENT_COL]].notna().any(axis=1)
    ]

    # Standarise names
    df.rename({
        Config.TICKET_SUMMARY_COL: "summary",
        Config.INTERACTION_CONTENT_COL: "content",
        "Ticket id": "ticket_id",
        "Interaction id": "interaction_id",
        "Type 1": "y1",
        "Type 2": "y2",
        "Type 3": "y3",
        "Type 4": "y4"
    }, axis=1, inplace=True)

    # Select important columns
    df = df[['ticket_id','interaction_id', "summary", "content", "y1", "y2", "y3", "y4"]]

    # fill
    df['summary'] = df['summary'].fillna(df['y4']) # .fillna(df['y3'])....

    # remove nan
    df = df[ df['content'].notna() ]

    # Rebuilds index
    df.reset_index(drop=True, inplace=True)

    return df

def pre_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    applies all preprocessing steps in order
    :param df: dataframe from get_input_data
    :return: pd.DataFrame
    """
    df = de_duplicate(df)
    df = remove_noise(df)
    return df

def prime_column(s: pd.Series, replace: t.Sequence[t.Tuple[str, str]]) -> pd.Series:
    """
    primes a series by lowercasing, replacing noise, and then stripping

    :param s:
    :param replace: array-like of tuple [(regex_pattern, replacement),...]
    :return: pd.Series
    """
    s = s.copy().str.lower()

    for (pattern, replacement) in replace:
        s = s.str.replace(pattern, replacement, regex=True)

    return s.str.strip()


def remove_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes noise from company-csv dataframe
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """

    df['summary'] = prime_column(df['summary'], [
        ("(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)", ' '),
        (r'\s+', ' ')
    ])

    df['content'] = prime_column(df['content'], [
        ("(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)", " "),
        ("(january|february|march|april|may|june|july|august|september|october|november|december)", " "),
        ("(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", " "),
        ("(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", " "),
        ("\d{2}(:|.)\d{2}", " "),
        ("(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))", " "),
        ("dear ((customer)|(user))", " "),
        ("dear", " "),
        ("(hello)|(hallo)|(hi )|(hi there)", " "),
        ("good morning", " "),
        ("thank you for your patience ((during (our)? investigation)|(and cooperation))?", " "),
        ("thank you for contacting us", " "),
        ("thank you for your availability", " "),
        ("thank you for providing us this information", " "),
        ("thank you for contacting", " "),
        ("thank you for reaching us (back)?", " "),
        ("thank you for patience", " "),
        ("thank you for (your)? reply", " "),
        ("thank you for (your)? response", " "),
        ("thank you for (your)? cooperation", " "),
        ("thank you for providing us with more information", " "),
        ("thank you very kindly", " "),
        ("thank you( very much)?", " "),
        ("i would like to follow up on the case you raised on the date", " "),
        ("i will do my very best to assist you", " "),
        ("in order to give you the best solution", " "),
        ("could you please clarify your request with following information:", " "),
        ("in this matter", " "),
        ("we hope you(( are)|('re)) doing ((fine)|(well))", " "),
        ("i would like to follow up on the case you raised on", " "),
        ("we apologize for the inconvenience", " "),
        ("sent from my huawei (cell )?phone", " "),
        ("original message", " "),
        ("customer support team", " "),
        ("(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
         " "),
        ("(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in", " "),
        ("canada, australia, new zealand and other countries", " "),
        ("\d+", " "),
        ("[^0-9a-zA-Z]+", " "),
        ("(\s|^).(\s|$)", " ")
    ])

    # drops in regards to value counts?? no thanks
    #good_y1 = df.y1.value_counts()[df.y1.value_counts() > 10].index
    #df = df.loc[df.y1.isin(good_y1)]

    return df


def de_duplicate(data: pd.DataFrame) -> pd.DataFrame:
    data["ic_deduplicated"] = ""

    cu_template = {
        "english":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?"]
        ,
        "german":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Ländern\.?"]
        ,
        "french":
            ["L'équipe d'assistance à la clientèle d'Aspiegel\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est une société de droit irlandais dont le siège est à Dublin, en Irlande\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est le fournisseur de services mobiles Huawei aux propriétaires d'appareils Huawei et Honor en Europe, au Canada, en Australie, en Nouvelle-Zélande et dans d'autres pays\.?"]
        ,
        "spanish":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) es una sociedad constituida en virtud de la legislación de Irlanda con su sede en Dublín, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios móviles de Huawei a los propietarios de dispositivos de Huawei y Honor en Europa, Canadá, Australia, Nueva Zelanda y otros países\.?"]
        ,
        "italian":
            ["Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\)),?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è una società costituita secondo le leggi irlandesi con sede a Dublino, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è il fornitore di servizi mobili Huawei per i proprietari di dispositivi Huawei e Honor in Europa, Canada, Australia, Nuova Zelanda e altri paesi\.?"]
        ,
        "portguese":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é uma empresa constituída segundo as leis da Irlanda, com sede em Dublin, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é o provedor de Huawei Mobile Services para Huawei e Honor proprietários de dispositivos na Europa, Canadá, Austrália, Nova Zelândia e outros países\.?"]
        ,
    }

    cu_pattern = ""
    for i in sum(list(cu_template.values()), []):
        cu_pattern = cu_pattern + f"({i})|"
    cu_pattern = cu_pattern[:-1]

    # -------- email split template

    pattern_1 = "(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)"
    pattern_2 = "(On.{30,60}wrote:)"
    pattern_3 = "(Re\s?:|RE\s?:)"
    pattern_4 = "(\*\*\*\*\*\(PERSON\) Support issue submit)"
    pattern_5 = "(\s?\*\*\*\*\*\(PHONE\))*$"

    split_pattern = f"{pattern_1}|{pattern_2}|{pattern_3}|{pattern_4}|{pattern_5}"

    # -------- start processing ticket data

    tickets = data["ticket_id"].value_counts()

    for t in tickets.index:
        #print(t)
        df = data.loc[data['ticket_id'] == t,]

        # for one ticket content data
        ic_set = set([])
        ic_deduplicated = []
        for ic in df["content"]:

            #print(ic)

            ic_r = re.split(split_pattern, ic)
            # ic_r = sum(ic_r, [])

            ic_r = [i for i in ic_r if i is not None]

            # replace split patterns
            ic_r = [re.sub(split_pattern, "", i.strip()) for i in ic_r]

            # replace customer template
            ic_r = [re.sub(cu_pattern, "", i.strip()) for i in ic_r]

            ic_current = []
            for i in ic_r:
                if len(i) > 0:
                    # print(i)
                    if i not in ic_set:
                        ic_set.add(i)
                        i = i + "\n"
                        ic_current = ic_current + [i]

            #print(ic_current)
            ic_deduplicated = ic_deduplicated + [' '.join(ic_current)]
        data.loc[data["ticket_id"] == t, "ic_deduplicated"] = ic_deduplicated
    data.to_csv('out.csv')
    data["content"] = data['ic_deduplicated']
    data = data.drop(columns=['ic_deduplicated'])
    return data