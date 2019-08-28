/*
 * Exception to throw in case an Instance does not have the class attribute 
 * as its last attribute
 */
package MScMartinoCode;

/**
 *
 * @author Martino Gonzales
 */
public class ClassIndexMismatchException extends Exception{
        public ClassIndexMismatchException(String errMessage) {
            super(errMessage);
        }
    
}
