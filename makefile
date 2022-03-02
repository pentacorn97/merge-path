CC = nvcc
DIR_SRC = ./src
DIR_OBJ = ./obj
DIR_INC = ./inc
FLAG = --allow-unsupported-compiler

SRC = $(wildcard $(DIR_SRC)/*.cu)
OBJ_ALL = $(patsubst %.cu, %.o, $(addprefix $(DIR_OBJ)/, $(notdir $(SRC))))
REBUILDABLE = ./*.exe ./*.exp ./*.lib ./*.obj $(OBJ_ALL)

clean:
	rm -f $(REBUILDABLE)

$(DIR_OBJ)/%.obj : $(DIR_SRC)/%.cu
	$(CC) $(FLAG) -o $@ -c $<
