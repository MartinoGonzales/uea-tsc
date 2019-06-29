/*
 * Exception to throw in case an Instance does not have the class attribute 
 * as its last attribute
 */
package MScMartinoCode;

/**
 *
 * @author fax14yxu
 */
public class ClassIndexMismatchException extends Exception{
        public ClassIndexMismatchException(String errMessage) {
            super(errMessage);
        }
    
}
