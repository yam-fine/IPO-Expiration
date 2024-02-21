# import AV_Import_Data
# import Financial_Modeling_Prep_Data
import SimFin_Import_Data as sf
import Simfin_daily_data as sfdd
import learningAlgs as lA
import parse_excel as pe


if __name__ == '__main__':
    # data = AV_Import_Data.AV_Import_Data().get_data()
    # data2 = Financial_Modeling_Prep_Data.FMP_Data().get_data()
    # data = sf.Simfin_data()
    pe.ParseExcel("ALLIPO.xlsx").save_excel("ALLIPO_new.xlsx")
    data = sfdd.Simfin_data("ALLIPO_new.xlsx")
    learningClass = lA.learningAlgs(data.X, data.y)
    model = learningClass.neuralNetBinClassification()
    model.predict("Predict.xlsx")
